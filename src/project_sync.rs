//! Git-like, object-store-backed synchronization for portable CueMap projects.
//!
//! Project packages and commit records are immutable. A small `HEAD.json` is
//! advanced with an S3 conditional write, preventing stale replicas from
//! silently overwriting one another.

use crate::multi_tenant::validate_project_id;
use crate::project_package;
use bytes::Bytes;
use object_store::aws::{AmazonS3Builder, S3ConditionalPut};
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, PutMode, PutPayload, UpdateVersion};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

const COMMIT_FORMAT: &str = "cuemap-sync-commit";
const STATE_FORMAT: &str = "cuemap-sync-state";
const SYNC_VERSION: u32 = 1;
const MAX_ANCESTRY_DEPTH: usize = 10_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SyncCommit {
    pub format: String,
    pub version: u32,
    pub project_id: String,
    pub generation: u64,
    pub commit_sha256: String,
    pub package_sha256: String,
    pub state_sha256: String,
    pub parent_commit_sha256: Option<String>,
    pub package_key: String,
    pub writer_id: String,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct LocalSyncState {
    format: String,
    version: u32,
    project_id: String,
    remote: String,
    generation: u64,
    commit_sha256: String,
    package_sha256: String,
    state_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SyncAction {
    Pushed,
    Pulled,
    UpToDate,
    Adopted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SyncResult {
    pub action: SyncAction,
    pub project_id: String,
    pub remote: String,
    pub generation: u64,
    pub commit_sha256: String,
    pub package_sha256: String,
}

#[derive(Debug)]
pub enum SyncRun {
    Complete(SyncResult),
    PullRequired(PreparedSyncPull),
}

#[derive(Debug)]
pub struct PreparedSyncPull {
    result: SyncResult,
    package_path: PathBuf,
    commit: SyncCommit,
    remote_uri: String,
    expected_local_state_sha256: Option<String>,
}

impl PreparedSyncPull {
    pub fn result(&self) -> &SyncResult {
        &self.result
    }
}

impl Drop for PreparedSyncPull {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.package_path);
    }
}

#[derive(Debug, Clone)]
struct RemoteCommit {
    commit: SyncCommit,
    e_tag: Option<String>,
    version: Option<String>,
}

struct S3SyncRemote {
    canonical_uri: String,
    bucket: String,
    prefix: String,
    store: Arc<dyn ObjectStore>,
}

#[derive(Debug, Deserialize)]
struct AwsProcessCredentials {
    #[serde(rename = "AccessKeyId")]
    access_key_id: String,
    #[serde(rename = "SecretAccessKey")]
    secret_access_key: String,
    #[serde(rename = "SessionToken")]
    session_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SyncDecision {
    Push { generation: u64 },
    Pull,
    UpToDate,
    Adopt,
    Diverged,
    Missing,
}

impl S3SyncRemote {
    fn new(uri: &str) -> Result<Self, String> {
        let (canonical_uri, bucket, prefix) = parse_remote_uri(uri)?;
        let credentials = export_aws_credentials()?;
        let region = resolve_bucket_region(&bucket)?;
        let mut builder = AmazonS3Builder::from_env()
            .with_bucket_name(&bucket)
            .with_region(region)
            .with_access_key_id(credentials.access_key_id)
            .with_secret_access_key(credentials.secret_access_key)
            .with_conditional_put(S3ConditionalPut::ETagMatch);
        if let Some(token) = credentials.session_token {
            builder = builder.with_token(token);
        }
        let store = builder
            .build()
            .map_err(|error| format!("Failed to configure S3 sync remote: {error}"))?;
        Ok(Self {
            canonical_uri,
            bucket,
            prefix,
            store: Arc::new(store),
        })
    }

    fn project_root(&self, project_id: &str) -> String {
        let suffix = format!(".cuemap-sync/v1/projects/{project_id}");
        if self.prefix.is_empty() {
            suffix
        } else {
            format!("{}/{suffix}", self.prefix)
        }
    }

    fn head_key(&self, project_id: &str) -> String {
        format!("{}/HEAD.json", self.project_root(project_id))
    }

    fn commit_key(&self, project_id: &str, commit_sha256: &str) -> String {
        format!(
            "{}/commits/{commit_sha256}.json",
            self.project_root(project_id)
        )
    }

    fn package_key(&self, project_id: &str, package_sha256: &str) -> String {
        format!(
            "{}/objects/{package_sha256}.cuemap",
            self.project_root(project_id)
        )
    }

    fn object_uri(&self, key: &str) -> String {
        format!("s3://{}/{}", self.bucket, key)
    }

    async fn head(&self, project_id: &str) -> Result<Option<RemoteCommit>, String> {
        self.read_commit(&self.head_key(project_id), true).await
    }

    async fn commit(
        &self,
        project_id: &str,
        commit_sha256: &str,
    ) -> Result<Option<RemoteCommit>, String> {
        self.read_commit(&self.commit_key(project_id, commit_sha256), false)
            .await
    }

    async fn read_commit(
        &self,
        key: &str,
        allow_missing: bool,
    ) -> Result<Option<RemoteCommit>, String> {
        let object_path = ObjectPath::from(key);
        let result = match self.store.get(&object_path).await {
            Ok(result) => result,
            Err(object_store::Error::NotFound { .. }) if allow_missing => return Ok(None),
            Err(object_store::Error::NotFound { .. }) => {
                return Err(format!("Sync history object '{key}' is missing"))
            }
            Err(error) => return Err(format!("Failed to read sync object '{key}': {error}")),
        };
        let e_tag = result.meta.e_tag.clone();
        let version = result.meta.version.clone();
        let bytes = result
            .bytes()
            .await
            .map_err(|error| format!("Failed to download sync object '{key}': {error}"))?;
        let commit: SyncCommit = serde_json::from_slice(&bytes)
            .map_err(|error| format!("Invalid sync commit '{key}': {error}"))?;
        validate_commit(&commit)?;
        let expected_package_key = self.package_key(&commit.project_id, &commit.package_sha256);
        if commit.package_key != expected_package_key {
            return Err(format!(
                "Sync commit '{}' references an unexpected package key",
                commit.commit_sha256
            ));
        }
        if key != self.head_key(&commit.project_id)
            && key != self.commit_key(&commit.project_id, &commit.commit_sha256)
        {
            return Err(format!(
                "Sync commit '{}' is stored under the wrong key",
                commit.commit_sha256
            ));
        }
        Ok(Some(RemoteCommit {
            commit,
            e_tag,
            version,
        }))
    }

    async fn is_descendant(
        &self,
        head: &SyncCommit,
        ancestor_commit_sha256: &str,
    ) -> Result<bool, String> {
        if head.commit_sha256 == ancestor_commit_sha256 {
            return Ok(true);
        }
        let mut current = head.clone();
        let mut seen = HashSet::new();
        for _ in 0..MAX_ANCESTRY_DEPTH {
            if !seen.insert(current.commit_sha256.clone()) {
                return Err("Sync history contains a cycle".to_string());
            }
            let Some(parent) = current.parent_commit_sha256.as_deref() else {
                return Ok(false);
            };
            if parent == ancestor_commit_sha256 {
                return Ok(true);
            }
            let next = self
                .commit(&current.project_id, parent)
                .await?
                .ok_or_else(|| format!("Sync history commit '{parent}' is missing"))?
                .commit;
            if next.project_id != current.project_id
                || next.generation.checked_add(1) != Some(current.generation)
            {
                return Err("Sync history has an invalid generation chain".to_string());
            }
            current = next;
        }
        Err(format!(
            "Sync history exceeds the maximum depth of {MAX_ANCESTRY_DEPTH}"
        ))
    }

    async fn put_commit(&self, commit: &SyncCommit) -> Result<(), String> {
        let key = self.commit_key(&commit.project_id, &commit.commit_sha256);
        let bytes = serde_json::to_vec(commit)
            .map_err(|error| format!("Failed to encode sync commit: {error}"))?;
        let path = ObjectPath::from(key.clone());
        match self
            .store
            .put_opts(
                &path,
                PutPayload::from_bytes(Bytes::from(bytes.clone())),
                PutMode::Create.into(),
            )
            .await
        {
            Ok(_) => Ok(()),
            Err(object_store::Error::AlreadyExists { .. }) => {
                let existing = self
                    .commit(&commit.project_id, &commit.commit_sha256)
                    .await?
                    .ok_or_else(|| format!("Sync commit '{key}' disappeared"))?;
                if existing.commit == *commit {
                    Ok(())
                } else {
                    Err(format!("Immutable sync commit collision at '{key}'"))
                }
            }
            Err(error) => Err(format!("Failed to publish sync commit '{key}': {error}")),
        }
    }

    async fn advance_head(
        &self,
        commit: &SyncCommit,
        expected: Option<&RemoteCommit>,
    ) -> Result<(), String> {
        let bytes = serde_json::to_vec(commit)
            .map_err(|error| format!("Failed to encode sync head: {error}"))?;
        let mode = match expected {
            Some(previous) => PutMode::Update(UpdateVersion {
                e_tag: previous.e_tag.clone(),
                version: previous.version.clone(),
            }),
            None => PutMode::Create,
        };
        let key = self.head_key(&commit.project_id);
        let result = self
            .store
            .put_opts(
                &ObjectPath::from(key),
                PutPayload::from_bytes(Bytes::from(bytes)),
                mode.into(),
            )
            .await;
        match result {
            Ok(_) => Ok(()),
            Err(
                object_store::Error::AlreadyExists { .. }
                | object_store::Error::Precondition { .. },
            ) => Err(
                "Remote head changed during sync; no remote state was overwritten. Run sync again"
                    .to_string(),
            ),
            Err(error) => Err(format!("Failed to advance remote sync head: {error}")),
        }
    }
}

pub async fn sync_project(
    data_dir: &Path,
    project_id: &str,
    remote_uri: &str,
    allow_local_replace: bool,
) -> Result<SyncRun, String> {
    if !validate_project_id(project_id) {
        return Err(format!("Invalid project ID '{project_id}'"));
    }
    fs::create_dir_all(data_dir)
        .map_err(|error| format!("Failed to create '{}': {error}", data_dir.display()))?;
    let remote_uri_owned = remote_uri.to_string();
    let remote = tokio::task::spawn_blocking(move || S3SyncRemote::new(&remote_uri_owned))
        .await
        .map_err(|error| format!("Sync remote setup task failed: {error}"))??;
    let local_state = load_local_state(data_dir, project_id)?;
    if let Some(state) = &local_state {
        if state.remote != remote.canonical_uri {
            return Err(format!(
                "Project '{project_id}' is already linked to {}; refusing to reuse its sync base for {}",
                state.remote, remote.canonical_uri
            ));
        }
    }

    let main_snapshot = data_dir
        .join("snapshots")
        .join(format!("{project_id}.bin"));
    let local_exists = main_snapshot.is_file();
    let local_hash = if local_exists {
        Some(project_package::project_state_sha256(data_dir, project_id)?)
    } else {
        None
    };
    let remote_head = remote.head(project_id).await?;
    if let Some(head) = &remote_head {
        if head.commit.project_id != project_id {
            return Err(format!(
                "Remote head belongs to project '{}', not '{project_id}'",
                head.commit.project_id
            ));
        }
    }

    let descendant = match (&local_state, &remote_head) {
        (Some(state), Some(head))
            if state.commit_sha256 != head.commit.commit_sha256 =>
        {
            Some(
                remote
                    .is_descendant(&head.commit, &state.commit_sha256)
                    .await?,
            )
        }
        _ => None,
    };
    let decision = decide_sync(
        local_hash.as_deref(),
        local_state.as_ref(),
        remote_head.as_ref().map(|head| &head.commit),
        descendant,
    );

    match decision {
        SyncDecision::Missing => Err(format!(
            "Project '{project_id}' does not exist locally and the remote has no head"
        )),
        SyncDecision::Diverged => Err(format!(
            "Project '{project_id}' has diverged from {}. No data was changed; restore one side to the common base or use a separate project/remote",
            remote.canonical_uri
        )),
        SyncDecision::UpToDate => {
            let head = remote_head.expect("up-to-date decision requires remote head");
            Ok(SyncRun::Complete(sync_result(
                SyncAction::UpToDate,
                &remote,
                &head.commit,
            )))
        }
        SyncDecision::Adopt => {
            let head = remote_head.expect("adopt decision requires remote head");
            save_local_state(data_dir, &remote, &head.commit)?;
            Ok(SyncRun::Complete(sync_result(
                SyncAction::Adopted,
                &remote,
                &head.commit,
            )))
        }
        SyncDecision::Push { generation } => {
            let state_sha256 = local_hash.expect("push decision requires local project");
            push_local_project(
                data_dir,
                project_id,
                &remote,
                remote_head.as_ref(),
                generation,
                state_sha256,
            )
            .await
        }
        SyncDecision::Pull => {
            let head = remote_head.expect("pull decision requires remote head");
            let prepared = prepare_remote_project(
                project_id,
                &remote,
                &head.commit,
                local_hash.clone(),
            )
            .await?;
            if local_exists && !allow_local_replace {
                return Ok(SyncRun::PullRequired(prepared));
            }
            let result = prepared.result.clone();
            complete_prepared_pull(data_dir, &prepared, local_exists)?;
            Ok(SyncRun::Complete(result))
        }
    }
}

async fn push_local_project(
    data_dir: &Path,
    project_id: &str,
    remote: &S3SyncRemote,
    previous: Option<&RemoteCommit>,
    generation: u64,
    state_sha256: String,
) -> Result<SyncRun, String> {
    let package_path = std::env::temp_dir().join(format!(
        "cuemap-sync-push-{project_id}-{}.cuemap",
        Uuid::new_v4()
    ));
    let operation = async {
        let pack_data_dir = data_dir.to_path_buf();
        let pack_project_id = project_id.to_string();
        let pack_path = package_path.clone();
        let (package_sha256, writer_id) = tokio::task::spawn_blocking(move || {
            project_package::pack_project(&pack_data_dir, &pack_project_id, &pack_path, false)?;
            Ok::<_, String>((
                project_package::file_sha256(&pack_path)?,
                writer_id(&pack_data_dir)?,
            ))
        })
        .await
        .map_err(|error| format!("Sync package task failed: {error}"))??;
        let package_key = remote.package_key(project_id, &package_sha256);
        let upload_path = package_path.clone();
        let upload_uri = remote.object_uri(&package_key);
        tokio::task::spawn_blocking(move || project_package::upload_s3(&upload_path, &upload_uri))
            .await
            .map_err(|error| format!("Sync upload task failed: {error}"))??;
        let mut commit = SyncCommit {
            format: COMMIT_FORMAT.to_string(),
            version: SYNC_VERSION,
            project_id: project_id.to_string(),
            generation,
            commit_sha256: String::new(),
            package_sha256,
            state_sha256,
            parent_commit_sha256: previous
                .map(|head| head.commit.commit_sha256.clone()),
            package_key,
            writer_id,
            created_at: unix_seconds(),
        };
        commit.commit_sha256 = calculate_commit_hash(&commit)?;
        remote.put_commit(&commit).await?;
        remote.advance_head(&commit, previous).await?;
        save_local_state(data_dir, remote, &commit)?;
        Ok::<SyncRun, String>(SyncRun::Complete(sync_result(
            SyncAction::Pushed,
            remote,
            &commit,
        )))
    }
    .await;
    let _ = fs::remove_file(package_path);
    operation
}

async fn prepare_remote_project(
    project_id: &str,
    remote: &S3SyncRemote,
    commit: &SyncCommit,
    expected_local_state_sha256: Option<String>,
) -> Result<PreparedSyncPull, String> {
    let package_path = std::env::temp_dir().join(format!(
        "cuemap-sync-pull-{project_id}-{}.cuemap",
        Uuid::new_v4()
    ));
    let download_uri = remote.object_uri(&commit.package_key);
    let validate_path = package_path.clone();
    let expected_package_sha256 = commit.package_sha256.clone();
    let expected_project_id = project_id.to_string();
    let operation = tokio::task::spawn_blocking(move || -> Result<(), String> {
        project_package::download_s3(&download_uri, &validate_path)?;
        let package_sha256 = project_package::file_sha256(&validate_path)?;
        if package_sha256 != expected_package_sha256 {
            return Err(format!(
                "Downloaded package hash mismatch: expected {}, got {package_sha256}",
                expected_package_sha256
            ));
        }
        let manifest = project_package::inspect_project_package(&validate_path)?;
        if manifest.project_id != expected_project_id {
            return Err(format!(
                "Downloaded package belongs to '{}', not '{}'",
                manifest.project_id, expected_project_id
            ));
        }
        Ok(())
    })
    .await
    .map_err(|error| format!("Sync download task failed: {error}"))?;
    if let Err(error) = operation {
        let _ = fs::remove_file(package_path);
        return Err(error);
    }
    Ok(PreparedSyncPull {
        result: sync_result(SyncAction::Pulled, remote, commit),
        package_path,
        commit: commit.clone(),
        remote_uri: remote.canonical_uri.clone(),
        expected_local_state_sha256,
    })
}

pub fn complete_prepared_pull(
    data_dir: &Path,
    prepared: &PreparedSyncPull,
    overwrite: bool,
) -> Result<(), String> {
    if overwrite {
        if let Some(expected) = prepared.expected_local_state_sha256.as_deref() {
            let current =
                project_package::project_state_sha256(data_dir, &prepared.commit.project_id)?;
            if current != expected {
                return Err(
                    "Local project changed while sync was preparing the pull; no local state was replaced. Run sync again"
                        .to_string(),
                );
            }
        }
    }
    project_package::load_project_package_with_state_hash(
        data_dir,
        &prepared.package_path,
        overwrite,
        &prepared.commit.state_sha256,
    )?;
    save_local_state_for_uri(data_dir, &prepared.remote_uri, &prepared.commit)
}

fn decide_sync(
    local_hash: Option<&str>,
    local_state: Option<&LocalSyncState>,
    remote_head: Option<&SyncCommit>,
    remote_descends_from_base: Option<bool>,
) -> SyncDecision {
    match (local_hash, remote_head) {
        (None, None) => SyncDecision::Missing,
        (Some(_), None) => SyncDecision::Push { generation: 1 },
        (None, Some(_)) => SyncDecision::Pull,
        (Some(local), Some(remote)) => match local_state {
            None if local == remote.state_sha256 => SyncDecision::Adopt,
            None => SyncDecision::Diverged,
            Some(base)
                if base.generation == remote.generation
                    && base.commit_sha256 == remote.commit_sha256 =>
            {
                if local == base.state_sha256 {
                    SyncDecision::UpToDate
                } else {
                    SyncDecision::Push {
                        generation: remote.generation.saturating_add(1),
                    }
                }
            }
            Some(_) if local == remote.state_sha256 => SyncDecision::Adopt,
            Some(base) if local == base.state_sha256 && remote_descends_from_base == Some(true) => {
                SyncDecision::Pull
            }
            Some(_) => SyncDecision::Diverged,
        },
    }
}

fn sync_result(action: SyncAction, remote: &S3SyncRemote, commit: &SyncCommit) -> SyncResult {
    SyncResult {
        action,
        project_id: commit.project_id.clone(),
        remote: remote.canonical_uri.clone(),
        generation: commit.generation,
        commit_sha256: commit.commit_sha256.clone(),
        package_sha256: commit.package_sha256.clone(),
    }
}

fn state_path(data_dir: &Path, project_id: &str) -> PathBuf {
    data_dir.join("sync").join(format!("{project_id}.json"))
}

fn load_local_state(data_dir: &Path, project_id: &str) -> Result<Option<LocalSyncState>, String> {
    let path = state_path(data_dir, project_id);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&path)
        .map_err(|error| format!("Failed to read sync state '{}': {error}", path.display()))?;
    let state: LocalSyncState = serde_json::from_slice(&bytes)
        .map_err(|error| format!("Invalid sync state '{}': {error}", path.display()))?;
    if state.format != STATE_FORMAT
        || state.version != SYNC_VERSION
        || state.project_id != project_id
        || !valid_hash(&state.commit_sha256)
        || !valid_hash(&state.package_sha256)
        || !valid_hash(&state.state_sha256)
    {
        return Err(format!("Invalid sync state '{}': unsupported or corrupt", path.display()));
    }
    Ok(Some(state))
}

fn save_local_state(
    data_dir: &Path,
    remote: &S3SyncRemote,
    commit: &SyncCommit,
) -> Result<(), String> {
    save_local_state_for_uri(data_dir, &remote.canonical_uri, commit)
}

fn save_local_state_for_uri(
    data_dir: &Path,
    remote_uri: &str,
    commit: &SyncCommit,
) -> Result<(), String> {
    let state = LocalSyncState {
        format: STATE_FORMAT.to_string(),
        version: SYNC_VERSION,
        project_id: commit.project_id.clone(),
        remote: remote_uri.to_string(),
        generation: commit.generation,
        commit_sha256: commit.commit_sha256.clone(),
        package_sha256: commit.package_sha256.clone(),
        state_sha256: commit.state_sha256.clone(),
    };
    let path = state_path(data_dir, &commit.project_id);
    let parent = path.parent().expect("sync state has a parent");
    fs::create_dir_all(parent)
        .map_err(|error| format!("Failed to create '{}': {error}", parent.display()))?;
    let temp = path.with_extension(format!("json.{}.tmp", Uuid::new_v4()));
    let bytes = serde_json::to_vec_pretty(&state)
        .map_err(|error| format!("Failed to encode sync state: {error}"))?;
    let result = (|| -> Result<(), String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|error| format!("Failed to create '{}': {error}", temp.display()))?;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|error| format!("Failed to persist sync state: {error}"))?;
        #[cfg(windows)]
        if path.exists() {
            fs::remove_file(&path)
                .map_err(|error| format!("Failed to replace '{}': {error}", path.display()))?;
        }
        fs::rename(&temp, &path)
            .map_err(|error| format!("Failed to finalize '{}': {error}", path.display()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}

fn writer_id(data_dir: &Path) -> Result<String, String> {
    let sync_dir = data_dir.join("sync");
    fs::create_dir_all(&sync_dir)
        .map_err(|error| format!("Failed to create '{}': {error}", sync_dir.display()))?;
    let path = sync_dir.join("writer-id");
    if let Ok(existing) = fs::read_to_string(&path) {
        let existing = existing.trim();
        if Uuid::parse_str(existing).is_ok() {
            return Ok(existing.to_string());
        }
        return Err(format!("Invalid sync writer ID in '{}'", path.display()));
    }
    let value = Uuid::new_v4().to_string();
    match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(mut file) => {
            file.write_all(value.as_bytes())
                .and_then(|_| file.sync_all())
                .map_err(|error| format!("Failed to save sync writer ID: {error}"))?;
            Ok(value)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = fs::read_to_string(&path)
                .map_err(|read_error| format!("Failed to read sync writer ID: {read_error}"))?;
            let existing = existing.trim();
            Uuid::parse_str(existing)
                .map_err(|_| format!("Invalid sync writer ID in '{}'", path.display()))?;
            Ok(existing.to_string())
        }
        Err(error) => Err(format!("Failed to create sync writer ID: {error}")),
    }
}

fn parse_remote_uri(value: &str) -> Result<(String, String, String), String> {
    project_package::validate_s3_uri(value, true)?;
    let rest = value.trim_end_matches('/').trim_start_matches("s3://");
    let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
    let prefix = prefix.trim_matches('/').to_string();
    let canonical = if prefix.is_empty() {
        format!("s3://{bucket}")
    } else {
        format!("s3://{bucket}/{prefix}")
    };
    Ok((canonical, bucket.to_string(), prefix))
}

fn validate_commit(commit: &SyncCommit) -> Result<(), String> {
    if commit.format != COMMIT_FORMAT
        || commit.version != SYNC_VERSION
        || !validate_project_id(&commit.project_id)
        || commit.generation == 0
        || !valid_hash(&commit.commit_sha256)
        || !valid_hash(&commit.package_sha256)
        || !valid_hash(&commit.state_sha256)
        || commit
            .parent_commit_sha256
            .as_deref()
            .is_some_and(|hash| !valid_hash(hash))
        || commit.package_key.starts_with('/')
        || commit.package_key.split('/').any(|part| part == "..")
        || Uuid::parse_str(&commit.writer_id).is_err()
    {
        return Err("Unsupported or corrupt CueMap sync commit".to_string());
    }
    if calculate_commit_hash(commit)? != commit.commit_sha256 {
        return Err("CueMap sync commit hash does not match its contents".to_string());
    }
    Ok(())
}

fn calculate_commit_hash(commit: &SyncCommit) -> Result<String, String> {
    let mut unhashed = commit.clone();
    unhashed.commit_sha256.clear();
    let bytes = serde_json::to_vec(&unhashed)
        .map_err(|error| format!("Failed to encode sync commit for hashing: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn export_aws_credentials() -> Result<AwsProcessCredentials, String> {
    let output = Command::new("aws")
        .args(["configure", "export-credentials", "--format", "process"])
        .output()
        .map_err(|error| format!("Failed to run AWS CLI credential export: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "AWS CLI credential export failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("AWS CLI returned invalid credential JSON: {error}"))
}

fn resolve_bucket_region(bucket: &str) -> Result<String, String> {
    let output = Command::new("aws")
        .args([
            "s3api",
            "get-bucket-location",
            "--bucket",
            bucket,
            "--query",
            "LocationConstraint",
            "--output",
            "text",
        ])
        .output()
        .map_err(|error| format!("Failed to resolve S3 bucket region: {error}"))?;
    if output.status.success() {
        let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
        return Ok(match value.as_str() {
            "" | "None" | "null" => "us-east-1".to_string(),
            "EU" => "eu-west-1".to_string(),
            _ => value,
        });
    }
    if let Ok(region) = std::env::var("AWS_REGION").or_else(|_| std::env::var("AWS_DEFAULT_REGION")) {
        if !region.trim().is_empty() {
            return Ok(region);
        }
    }
    let configured = Command::new("aws")
        .args(["configure", "get", "region"])
        .output()
        .map_err(|error| format!("Failed to read AWS CLI region: {error}"))?;
    if configured.status.success() {
        let region = String::from_utf8_lossy(&configured.stdout).trim().to_string();
        if !region.is_empty() {
            return Ok(region);
        }
    }
    Err(format!(
        "Could not resolve the AWS region for bucket '{bucket}': {}",
        String::from_utf8_lossy(&output.stderr).trim()
    ))
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(value: char) -> String {
        value.to_string().repeat(64)
    }

    fn commit(generation: u64, package: char, state: char) -> SyncCommit {
        let mut commit = SyncCommit {
            format: COMMIT_FORMAT.to_string(),
            version: SYNC_VERSION,
            project_id: "sync-project".to_string(),
            generation,
            commit_sha256: String::new(),
            package_sha256: hash(package),
            state_sha256: hash(state),
            parent_commit_sha256: None,
            package_key: format!("objects/{}.cuemap", hash(package)),
            writer_id: Uuid::nil().to_string(),
            created_at: 1,
        };
        commit.commit_sha256 = calculate_commit_hash(&commit).unwrap();
        commit
    }

    fn state(commit: &SyncCommit) -> LocalSyncState {
        LocalSyncState {
            format: STATE_FORMAT.to_string(),
            version: SYNC_VERSION,
            project_id: "sync-project".to_string(),
            remote: "s3://bucket/team".to_string(),
            generation: commit.generation,
            commit_sha256: commit.commit_sha256.clone(),
            package_sha256: commit.package_sha256.clone(),
            state_sha256: commit.state_sha256.clone(),
        }
    }

    #[test]
    fn sync_decision_fast_forwards_and_rejects_divergence() {
        let current = commit(1, 'a', '1');
        let base = state(&current);
        assert_eq!(
            decide_sync(Some(&hash('1')), Some(&base), Some(&current), None),
            SyncDecision::UpToDate
        );
        assert_eq!(
            decide_sync(Some(&hash('2')), Some(&base), Some(&current), None),
            SyncDecision::Push { generation: 2 }
        );

        let advanced = commit(2, 'b', '2');
        assert_eq!(
            decide_sync(Some(&hash('1')), Some(&base), Some(&advanced), Some(true)),
            SyncDecision::Pull
        );
        assert_eq!(
            decide_sync(Some(&hash('3')), Some(&base), Some(&advanced), Some(true)),
            SyncDecision::Diverged
        );
        assert_eq!(
            decide_sync(Some(&hash('1')), Some(&base), Some(&advanced), Some(false)),
            SyncDecision::Diverged
        );
    }

    #[test]
    fn first_sync_adopts_only_identical_remote_state() {
        let remote = commit(4, 'd', '7');
        assert_eq!(
            decide_sync(Some(&hash('7')), None, Some(&remote), None),
            SyncDecision::Adopt
        );
        assert_eq!(
            decide_sync(Some(&hash('8')), None, Some(&remote), None),
            SyncDecision::Diverged
        );
        assert_eq!(
            decide_sync(None, None, Some(&remote), None),
            SyncDecision::Pull
        );
        assert_eq!(
            decide_sync(Some(&hash('8')), None, None, None),
            SyncDecision::Push { generation: 1 }
        );
    }

    #[test]
    fn commit_hash_detects_history_tampering() {
        let mut value = commit(3, 'c', '4');
        assert!(validate_commit(&value).is_ok());
        value.generation = 4;
        assert!(validate_commit(&value).is_err());
    }

    #[test]
    fn prepared_pull_refuses_a_local_project_changed_after_decision() {
        use crate::config::TuningConfig;
        use crate::multi_tenant::MultiTenantEngine;
        use crate::structures::MainStats;

        let data_dir = tempfile::tempdir().unwrap();
        let snapshots = data_dir.path().join("snapshots");
        fs::create_dir_all(&snapshots).unwrap();
        let engine = MultiTenantEngine::with_snapshots_dir(&snapshots, TuningConfig::default());
        let project_id = "sync-stale-local".to_string();
        let context = engine.get_or_create_project(project_id.clone()).unwrap();
        context.main.add_memory(
            "changed locally".to_string(),
            vec!["changed".to_string()],
            None,
            MainStats::default(),
            false,
        );
        drop(context);
        engine.save_project(&project_id).unwrap();

        let mut remote_commit = commit(2, 'b', '2');
        remote_commit.project_id = project_id;
        remote_commit.commit_sha256 = calculate_commit_hash(&remote_commit).unwrap();
        let prepared = PreparedSyncPull {
            result: SyncResult {
                action: SyncAction::Pulled,
                project_id: remote_commit.project_id.clone(),
                remote: "s3://bucket/team".to_string(),
                generation: remote_commit.generation,
                commit_sha256: remote_commit.commit_sha256.clone(),
                package_sha256: remote_commit.package_sha256.clone(),
            },
            package_path: data_dir.path().join("unused.cuemap"),
            commit: remote_commit,
            remote_uri: "s3://bucket/team".to_string(),
            expected_local_state_sha256: Some(hash('f')),
        };
        let error = complete_prepared_pull(data_dir.path(), &prepared, true).unwrap_err();
        assert!(error.contains("changed while sync"));
    }

    #[test]
    fn remote_uri_is_canonicalized() {
        assert_eq!(
            parse_remote_uri("s3://bucket/team/").unwrap(),
            (
                "s3://bucket/team".to_string(),
                "bucket".to_string(),
                "team".to_string()
            )
        );
    }
}
