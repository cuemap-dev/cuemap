//! Portable, checksummed CueMap project packages.
//!
//! A `.cuemap` file contains the already-built project snapshots plus any
//! disk-backed memory contents and optional CueBridge artifacts. Loading a
//! package installs those files directly; it does not replay ingestion.

use crate::multi_tenant::validate_project_id;
use crate::persistence::PersistenceManager;
use crate::structures::{LexiconStats, MainStats, MemoryId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;
use walkdir::WalkDir;

const PACKAGE_MAGIC: &[u8; 8] = b"CUEMAP01";
const PACKAGE_FORMAT: &str = "cuemap-project";
const PACKAGE_VERSION: u32 = 1;
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_PACKAGE_FILES: usize = 1_000_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectPackageFile {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectPackageManifest {
    pub format: String,
    pub version: u32,
    pub engine_version: String,
    pub project_id: String,
    pub created_at: u64,
    pub files: Vec<ProjectPackageFile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectPackageSummary {
    pub project_id: String,
    pub path: PathBuf,
    pub file_count: usize,
    pub size_bytes: u64,
}

#[derive(Debug)]
struct SourceFile {
    logical_path: String,
    source_path: PathBuf,
    size_bytes: u64,
    sha256: String,
}

/// Create one portable project package from a CueMap data directory.
pub fn pack_project(
    data_dir: &Path,
    project_id: &str,
    output_path: &Path,
    overwrite: bool,
) -> Result<ProjectPackageSummary, String> {
    validate_id(project_id)?;
    if output_path.exists() && !overwrite {
        return Err(format!(
            "Package '{}' already exists; pass --force to replace it",
            output_path.display()
        ));
    }

    let sources = collect_project_files(data_dir, project_id)?;
    let files = sources
        .iter()
        .map(|source| ProjectPackageFile {
            path: source.logical_path.clone(),
            size_bytes: source.size_bytes,
            sha256: source.sha256.clone(),
        })
        .collect();
    let manifest = ProjectPackageManifest {
        format: PACKAGE_FORMAT.to_string(),
        version: PACKAGE_VERSION,
        engine_version: env!("CARGO_PKG_VERSION").to_string(),
        project_id: project_id.to_string(),
        created_at: unix_seconds(),
        files,
    };
    let manifest_bytes = serde_json::to_vec(&manifest)
        .map_err(|error| format!("Failed to encode package manifest: {error}"))?;
    if manifest_bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err("Package manifest is too large".to_string());
    }

    if let Some(parent) = output_path.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Failed to create '{}': {error}", parent.display()))?;
    }
    let temp_path = package_temp_path(output_path);
    let write_result = (|| -> Result<(), String> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .map_err(|error| {
                format!(
                    "Failed to create temporary package '{}': {error}",
                    temp_path.display()
                )
            })?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(PACKAGE_MAGIC)
            .and_then(|_| writer.write_all(&(manifest_bytes.len() as u64).to_le_bytes()))
            .and_then(|_| writer.write_all(&manifest_bytes))
            .map_err(|error| format!("Failed to write package header: {error}"))?;

        for source in &sources {
            let mut input = BufReader::new(File::open(&source.source_path).map_err(|error| {
                format!("Failed to open '{}': {error}", source.source_path.display())
            })?);
            io::copy(&mut input, &mut writer).map_err(|error| {
                format!("Failed to package '{}': {error}", source.source_path.display())
            })?;
        }
        writer
            .flush()
            .map_err(|error| format!("Failed to flush package: {error}"))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| format!("Failed to sync package: {error}"))?;
        Ok(())
    })();

    if let Err(error) = write_result {
        let _ = fs::remove_file(&temp_path);
        return Err(error);
    }
    if overwrite && output_path.exists() {
        fs::remove_file(output_path).map_err(|error| {
            format!("Failed to replace '{}': {error}", output_path.display())
        })?;
    }
    fs::rename(&temp_path, output_path).map_err(|error| {
        let _ = fs::remove_file(&temp_path);
        format!(
            "Failed to finalize package '{}': {error}",
            output_path.display()
        )
    })?;

    let size_bytes = fs::metadata(output_path)
        .map_err(|error| format!("Failed to inspect package: {error}"))?
        .len();
    Ok(ProjectPackageSummary {
        project_id: project_id.to_string(),
        path: output_path.to_path_buf(),
        file_count: sources.len(),
        size_bytes,
    })
}

/// Read and validate only the package header and manifest.
pub fn inspect_project_package(package_path: &Path) -> Result<ProjectPackageManifest, String> {
    let file = File::open(package_path)
        .map_err(|error| format!("Failed to open '{}': {error}", package_path.display()))?;
    let mut reader = BufReader::new(file);
    let manifest = read_manifest(&mut reader)?;
    validate_manifest(&manifest)?;
    let expected = package_payload_offset(&manifest)?
        .checked_add(total_payload_bytes(&manifest)?)
        .ok_or_else(|| "Package size overflow".to_string())?;
    let actual = fs::metadata(package_path)
        .map_err(|error| format!("Failed to inspect '{}': {error}", package_path.display()))?
        .len();
    if expected != actual {
        return Err(format!(
            "Package length mismatch: manifest expects {expected} bytes, file has {actual}"
        ));
    }
    Ok(manifest)
}

/// Install a `.cuemap` package into a local CueMap data directory.
///
/// The caller must not replace a project that is currently loaded by a running
/// server. New projects may be installed and then demand-loaded by the server.
pub fn load_project_package(
    data_dir: &Path,
    package_path: &Path,
    overwrite: bool,
) -> Result<ProjectPackageSummary, String> {
    load_project_package_checked(data_dir, package_path, overwrite, None)
}

/// Install a package only when its normalized queryable state matches the
/// expected sync commit. The check happens in staging before any local file is
/// replaced.
pub fn load_project_package_with_state_hash(
    data_dir: &Path,
    package_path: &Path,
    overwrite: bool,
    expected_state_sha256: &str,
) -> Result<ProjectPackageSummary, String> {
    load_project_package_checked(
        data_dir,
        package_path,
        overwrite,
        Some(expected_state_sha256),
    )
}

fn load_project_package_checked(
    data_dir: &Path,
    package_path: &Path,
    overwrite: bool,
    expected_state_sha256: Option<&str>,
) -> Result<ProjectPackageSummary, String> {
    fs::create_dir_all(data_dir)
        .map_err(|error| format!("Failed to create '{}': {error}", data_dir.display()))?;
    let package_file = File::open(package_path)
        .map_err(|error| format!("Failed to open '{}': {error}", package_path.display()))?;
    let package_size = package_file
        .metadata()
        .map_err(|error| format!("Failed to inspect package: {error}"))?
        .len();
    let mut reader = BufReader::new(package_file);
    let manifest = read_manifest(&mut reader)?;
    validate_manifest(&manifest)?;

    let expected = package_payload_offset(&manifest)?
        .checked_add(total_payload_bytes(&manifest)?)
        .ok_or_else(|| "Package size overflow".to_string())?;
    if expected != package_size {
        return Err(format!(
            "Package length mismatch: manifest expects {expected} bytes, file has {package_size}"
        ));
    }

    let import_root = data_dir
        .join(".imports")
        .join(format!("{}-{}", manifest.project_id, Uuid::new_v4()));
    fs::create_dir_all(&import_root).map_err(|error| {
        format!(
            "Failed to create import staging directory '{}': {error}",
            import_root.display()
        )
    })?;

    let stage_result = (|| -> Result<(), String> {
        for entry in &manifest.files {
            let destination = import_root.join(logical_path(&entry.path)?);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent).map_err(|error| {
                    format!("Failed to create '{}': {error}", parent.display())
                })?;
            }
            let output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&destination)
                .map_err(|error| {
                    format!("Failed to create '{}': {error}", destination.display())
                })?;
            copy_exact_and_verify(&mut reader, output, entry)?;
        }
        let mut trailing = [0_u8; 1];
        if reader
            .read(&mut trailing)
            .map_err(|error| format!("Failed to finish reading package: {error}"))?
            != 0
        {
            return Err("Package contains trailing bytes".to_string());
        }
        validate_staged_project(&import_root)?;
        if let Some(expected) = expected_state_sha256 {
            let actual = staged_project_state_sha256(&import_root)?;
            if actual != expected {
                return Err(format!(
                    "Package project state hash mismatch: expected {expected}, got {actual}"
                ));
            }
        }
        install_staged_project(data_dir, &import_root, &manifest.project_id, overwrite)
    })();

    let _ = fs::remove_dir_all(&import_root);
    if let Some(imports) = import_root.parent() {
        let _ = fs::remove_dir(imports);
    }
    stage_result?;

    Ok(ProjectPackageSummary {
        project_id: manifest.project_id,
        path: package_path.to_path_buf(),
        file_count: manifest.files.len(),
        size_bytes: package_size,
    })
}

/// Resolve either an S3 bucket/prefix or a complete object URI for a project.
pub fn s3_destination(value: &str, project: &str) -> Result<String, String> {
    validate_s3_uri(value, true)?;
    if value.ends_with('/') {
        Ok(format!("{value}{project}.cuemap"))
    } else if value.trim_start_matches("s3://").contains('/') {
        Ok(value.to_string())
    } else {
        Ok(format!("{value}/{project}.cuemap"))
    }
}

/// Validate an S3 URI before handing it to the AWS CLI as one argument.
pub fn validate_s3_uri(value: &str, allow_bucket_only: bool) -> Result<(), String> {
    if value.chars().any(char::is_control) {
        return Err("S3 URI contains control characters".to_string());
    }
    let rest = value
        .strip_prefix("s3://")
        .ok_or_else(|| "Expected an s3:// URI".to_string())?;
    let (bucket, key) = rest.split_once('/').unwrap_or((rest, ""));
    if bucket.is_empty() || bucket == "." || bucket == ".." {
        return Err("S3 URI must include a bucket".to_string());
    }
    if !allow_bucket_only && key.trim_matches('/').is_empty() {
        return Err("S3 URI must include an object key".to_string());
    }
    Ok(())
}

/// Upload a package with the caller's configured AWS CLI credentials.
pub fn upload_s3(source: &Path, destination: &str) -> Result<(), String> {
    validate_s3_uri(destination, false)?;
    let output = std::process::Command::new("aws")
        .args(["s3", "cp"])
        .arg(source)
        .arg(destination)
        .args(["--only-show-errors", "--no-progress"])
        .output()
        .map_err(|error| format!("Failed to run AWS CLI: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "AWS CLI upload failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

/// Download a package with the caller's configured AWS CLI credentials.
pub fn download_s3(source: &str, destination: &Path) -> Result<(), String> {
    validate_s3_uri(source, false)?;
    let output = std::process::Command::new("aws")
        .args(["s3", "cp", source])
        .arg(destination)
        .args(["--only-show-errors", "--no-progress"])
        .output()
        .map_err(|error| format!("Failed to run AWS CLI: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "AWS CLI download failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn collect_project_files(data_dir: &Path, project_id: &str) -> Result<Vec<SourceFile>, String> {
    let snapshots = data_dir.join("snapshots");
    let candidates = [
        ("snapshot/main.bin".to_string(), snapshots.join(format!("{project_id}.bin"))),
        (
            "snapshot/aliases.bin".to_string(),
            snapshots.join(format!("{project_id}_aliases.bin")),
        ),
        (
            "snapshot/lexicon.bin".to_string(),
            snapshots.join(format!("{project_id}_lexicon.bin")),
        ),
    ];
    if !candidates[0].1.is_file() {
        return Err(format!(
            "Project snapshot '{}' was not found; save the project before packing",
            candidates[0].1.display()
        ));
    }

    let mut paths: Vec<(String, PathBuf)> = candidates
        .into_iter()
        .filter(|(_, path)| path.is_file())
        .collect();
    collect_directory(
        &data_dir.join("contents").join(project_id),
        "contents",
        &mut paths,
    )?;
    collect_directory(
        &data_dir.join("artifacts").join(project_id),
        "artifacts",
        &mut paths,
    )?;
    paths.sort_by(|left, right| left.0.cmp(&right.0));

    paths
        .into_iter()
        .map(|(logical_path, source_path)| {
            let size_bytes = fs::metadata(&source_path)
                .map_err(|error| format!("Failed to inspect '{}': {error}", source_path.display()))?
                .len();
            let sha256 = file_sha256(&source_path)?;
            Ok(SourceFile {
                logical_path,
                source_path,
                size_bytes,
                sha256,
            })
        })
        .collect()
}

fn collect_directory(
    root: &Path,
    logical_root: &str,
    paths: &mut Vec<(String, PathBuf)>,
) -> Result<(), String> {
    if !root.exists() {
        return Ok(());
    }
    for entry in WalkDir::new(root).follow_links(false).into_iter() {
        let entry = entry.map_err(|error| format!("Failed to walk '{}': {error}", root.display()))?;
        if entry.file_type().is_symlink() {
            return Err(format!(
                "Refusing to package symlink '{}'",
                entry.path().display()
            ));
        }
        if !entry.file_type().is_file() {
            continue;
        }
        let relative = entry
            .path()
            .strip_prefix(root)
            .map_err(|error| format!("Failed to resolve package path: {error}"))?;
        let relative = portable_relative_path(relative)?;
        paths.push((format!("{logical_root}/{relative}"), entry.path().to_path_buf()));
    }
    Ok(())
}

fn read_manifest(reader: &mut impl Read) -> Result<ProjectPackageManifest, String> {
    let mut magic = [0_u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|error| format!("Failed to read package header: {error}"))?;
    if &magic != PACKAGE_MAGIC {
        return Err("Not a CueMap project package".to_string());
    }
    let mut length = [0_u8; 8];
    reader
        .read_exact(&mut length)
        .map_err(|error| format!("Failed to read package manifest length: {error}"))?;
    let manifest_len = u64::from_le_bytes(length);
    if manifest_len == 0 || manifest_len > MAX_MANIFEST_BYTES {
        return Err(format!("Invalid package manifest length: {manifest_len}"));
    }
    let manifest_len: usize = manifest_len
        .try_into()
        .map_err(|_| "Package manifest is too large for this platform".to_string())?;
    let mut manifest_bytes = vec![0_u8; manifest_len];
    reader
        .read_exact(&mut manifest_bytes)
        .map_err(|error| format!("Failed to read package manifest: {error}"))?;
    serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("Invalid package manifest: {error}"))
}

fn validate_manifest(manifest: &ProjectPackageManifest) -> Result<(), String> {
    if manifest.format != PACKAGE_FORMAT {
        return Err(format!("Unsupported package format '{}'", manifest.format));
    }
    if manifest.version != PACKAGE_VERSION {
        return Err(format!(
            "Unsupported package version {} (expected {})",
            manifest.version, PACKAGE_VERSION
        ));
    }
    validate_id(&manifest.project_id)?;
    if manifest.files.is_empty() || manifest.files.len() > MAX_PACKAGE_FILES {
        return Err("Package contains an invalid number of files".to_string());
    }
    let mut seen = HashSet::new();
    let mut has_main = false;
    for entry in &manifest.files {
        logical_path(&entry.path)?;
        if !seen.insert(entry.path.as_str()) {
            return Err(format!("Package contains duplicate path '{}'", entry.path));
        }
        if entry.path == "snapshot/main.bin" {
            has_main = true;
        }
        if entry.sha256.len() != 64
            || !entry.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(format!("Invalid checksum for '{}'", entry.path));
        }
    }
    if !has_main {
        return Err("Package does not contain snapshot/main.bin".to_string());
    }
    Ok(())
}

fn logical_path(value: &str) -> Result<PathBuf, String> {
    if value.is_empty() || value.len() > 4096 || value.contains('\\') {
        return Err(format!("Unsafe package path '{value}'"));
    }
    let path = Path::new(value);
    if path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(format!("Unsafe package path '{value}'"));
    }
    let allowed = value == "snapshot/main.bin"
        || value == "snapshot/aliases.bin"
        || value == "snapshot/lexicon.bin"
        || value.starts_with("contents/")
        || value.starts_with("artifacts/");
    if !allowed {
        return Err(format!("Unsupported package path '{value}'"));
    }
    Ok(path.to_path_buf())
}

fn portable_relative_path(path: &Path) -> Result<String, String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(value) => parts.push(
                value
                    .to_str()
                    .ok_or_else(|| format!("Non-UTF-8 package path '{}'", path.display()))?,
            ),
            _ => return Err(format!("Unsafe package path '{}'", path.display())),
        }
    }
    if parts.is_empty() {
        return Err("Package file path cannot be empty".to_string());
    }
    Ok(parts.join("/"))
}

fn copy_exact_and_verify(
    reader: &mut impl Read,
    mut output: File,
    entry: &ProjectPackageFile,
) -> Result<(), String> {
    let mut remaining = entry.size_bytes;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    while remaining > 0 {
        let wanted = usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
        let read = reader
            .read(&mut buffer[..wanted])
            .map_err(|error| format!("Failed to read '{}': {error}", entry.path))?;
        if read == 0 {
            return Err(format!("Package ended while reading '{}'", entry.path));
        }
        output
            .write_all(&buffer[..read])
            .map_err(|error| format!("Failed to extract '{}': {error}", entry.path))?;
        hasher.update(&buffer[..read]);
        remaining -= read as u64;
    }
    output
        .sync_all()
        .map_err(|error| format!("Failed to sync '{}': {error}", entry.path))?;
    let actual = hex::encode(hasher.finalize());
    if actual != entry.sha256.to_ascii_lowercase() {
        return Err(format!("Checksum mismatch for '{}'", entry.path));
    }
    Ok(())
}

fn validate_staged_project(stage: &Path) -> Result<(), String> {
    let main_path = stage.join("snapshot/main.bin");
    let (memories, _, _, _, _) = PersistenceManager::load_from_path::<MainStats>(&main_path)
        .map_err(|error| format!("Invalid main snapshot: {error}"))?;
    let aliases = stage.join("snapshot/aliases.bin");
    if aliases.exists() {
        PersistenceManager::load_from_path::<MainStats>(&aliases)
            .map_err(|error| format!("Invalid aliases snapshot: {error}"))?;
    }
    let lexicon = stage.join("snapshot/lexicon.bin");
    if lexicon.exists() {
        PersistenceManager::load_from_path::<LexiconStats>(&lexicon)
            .map_err(|error| format!("Invalid lexicon snapshot: {error}"))?;
    }

    let missing_contents: Vec<MemoryId> = memories
        .iter()
        .filter_map(|entry| {
            let memory = entry.value();
            if memory.disk_backed
                && !stage
                    .join("contents")
                    .join(format!("{}.bin", memory.id))
                    .is_file()
            {
                Some(memory.id)
            } else {
                None
            }
        })
        .take(10)
        .collect();
    if !missing_contents.is_empty() {
        return Err(format!(
            "Package is missing disk-backed content for memories {:?}",
            missing_contents
        ));
    }
    Ok(())
}

fn install_staged_project(
    data_dir: &Path,
    stage: &Path,
    project_id: &str,
    overwrite: bool,
) -> Result<(), String> {
    let snapshots = data_dir.join("snapshots");
    fs::create_dir_all(&snapshots)
        .map_err(|error| format!("Failed to create '{}': {error}", snapshots.display()))?;
    let targets = [
        (stage.join("snapshot/main.bin"), snapshots.join(format!("{project_id}.bin"))),
        (
            stage.join("snapshot/aliases.bin"),
            snapshots.join(format!("{project_id}_aliases.bin")),
        ),
        (
            stage.join("snapshot/lexicon.bin"),
            snapshots.join(format!("{project_id}_lexicon.bin")),
        ),
        (
            stage.join("contents"),
            data_dir.join("contents").join(project_id),
        ),
        (
            stage.join("artifacts"),
            data_dir.join("artifacts").join(project_id),
        ),
    ];
    let conflicts: Vec<String> = targets
        .iter()
        .filter(|(_, target)| target.exists())
        .map(|(_, target)| target.display().to_string())
        .collect();
    if !conflicts.is_empty() && !overwrite {
        return Err(format!(
            "Project '{}' already exists at {}; pass --force to replace it",
            project_id,
            conflicts.join(", ")
        ));
    }

    let backup = data_dir
        .join(".imports")
        .join(format!("backup-{project_id}-{}", Uuid::new_v4()));
    fs::create_dir_all(&backup)
        .map_err(|error| format!("Failed to create rollback directory: {error}"))?;
    let mut backed_up = Vec::new();
    let mut installed = Vec::new();

    let install_result = (|| -> Result<(), String> {
        for (index, (_, target)) in targets.iter().enumerate() {
            if !target.exists() {
                continue;
            }
            let backup_target = backup.join(index.to_string());
            fs::rename(target, &backup_target).map_err(|error| {
                format!("Failed to stage existing '{}': {error}", target.display())
            })?;
            backed_up.push((backup_target, target.clone()));
        }

        // Install the main snapshot last so incomplete installs are not
        // discoverable as valid projects during normal startup scanning.
        for index in [1_usize, 2, 3, 4, 0] {
            let (source, target) = &targets[index];
            if !source.exists() {
                continue;
            }
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).map_err(|error| {
                    format!("Failed to create '{}': {error}", parent.display())
                })?;
            }
            fs::rename(source, target).map_err(|error| {
                format!("Failed to install '{}': {error}", target.display())
            })?;
            installed.push(target.clone());
        }
        Ok(())
    })();

    if let Err(error) = install_result {
        for target in installed.iter().rev() {
            remove_path(target);
        }
        for (source, target) in backed_up.into_iter().rev() {
            let _ = fs::rename(source, target);
        }
        let _ = fs::remove_dir_all(&backup);
        return Err(error);
    }
    fs::remove_dir_all(&backup)
        .map_err(|error| format!("Project installed but rollback cleanup failed: {error}"))?;
    Ok(())
}

fn remove_path(path: &Path) {
    if path.is_dir() {
        let _ = fs::remove_dir_all(path);
    } else {
        let _ = fs::remove_file(path);
    }
}

/// Hash a file without buffering it in memory.
pub fn file_sha256(path: &Path) -> Result<String, String> {
    let mut input = BufReader::new(
        File::open(path).map_err(|error| format!("Failed to open '{}': {error}", path.display()))?,
    );
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = input
            .read(&mut buffer)
            .map_err(|error| format!("Failed to hash '{}': {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Compute a stable hash of the queryable project state.
///
/// Snapshot timestamps and JSON object iteration order are normalized so an
/// otherwise unchanged project does not appear dirty merely because it was
/// saved again. Disk-backed contents and CueBridge artifacts are included.
pub fn project_state_sha256(data_dir: &Path, project_id: &str) -> Result<String, String> {
    validate_id(project_id)?;
    let snapshots = data_dir.join("snapshots");
    let snapshot_files = [
        ("snapshot/main.bin", snapshots.join(format!("{project_id}.bin"))),
        (
            "snapshot/aliases.bin",
            snapshots.join(format!("{project_id}_aliases.bin")),
        ),
        (
            "snapshot/lexicon.bin",
            snapshots.join(format!("{project_id}_lexicon.bin")),
        ),
    ];
    if !snapshot_files[0].1.is_file() {
        return Err(format!("Project snapshot for '{project_id}' was not found"));
    }

    let mut hasher = Sha256::new();
    for (logical_path, path) in snapshot_files {
        if path.is_file() {
            let bytes = normalized_snapshot_bytes(&path)?;
            hash_component(&mut hasher, logical_path.as_bytes(), &bytes);
        }
    }

    let mut extras = Vec::new();
    collect_directory(
        &data_dir.join("contents").join(project_id),
        "contents",
        &mut extras,
    )?;
    collect_directory(
        &data_dir.join("artifacts").join(project_id),
        "artifacts",
        &mut extras,
    )?;
    extras.sort_by(|left, right| left.0.cmp(&right.0));
    for (logical_path, path) in extras {
        let bytes = fs::read(&path)
            .map_err(|error| format!("Failed to read '{}': {error}", path.display()))?;
        hash_component(&mut hasher, logical_path.as_bytes(), &bytes);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn staged_project_state_sha256(import_root: &Path) -> Result<String, String> {
    let snapshot_files = [
        ("snapshot/main.bin", import_root.join("snapshot/main.bin")),
        (
            "snapshot/aliases.bin",
            import_root.join("snapshot/aliases.bin"),
        ),
        (
            "snapshot/lexicon.bin",
            import_root.join("snapshot/lexicon.bin"),
        ),
    ];
    let mut hasher = Sha256::new();
    for (logical_path, path) in snapshot_files {
        if path.is_file() {
            let bytes = normalized_snapshot_bytes(&path)?;
            hash_component(&mut hasher, logical_path.as_bytes(), &bytes);
        }
    }

    let mut extras = Vec::new();
    collect_directory(&import_root.join("contents"), "contents", &mut extras)?;
    collect_directory(&import_root.join("artifacts"), "artifacts", &mut extras)?;
    extras.sort_by(|left, right| left.0.cmp(&right.0));
    for (logical_path, path) in extras {
        let bytes = fs::read(&path)
            .map_err(|error| format!("Failed to read '{}': {error}", path.display()))?;
        hash_component(&mut hasher, logical_path.as_bytes(), &bytes);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn normalized_snapshot_bytes(path: &Path) -> Result<Vec<u8>, String> {
    let raw = fs::read(path)
        .map_err(|error| format!("Failed to read snapshot '{}': {error}", path.display()))?;
    let decoded = if crate::crypto::is_compressed(&raw) {
        zstd::stream::decode_all(io::Cursor::new(raw)).map_err(|error| {
            format!("Failed to decode snapshot '{}': {error}", path.display())
        })?
    } else {
        raw
    };
    let mut value: serde_json::Value = match serde_json::from_slice(&decoded) {
        Ok(value) => value,
        Err(_) => return Ok(decoded),
    };
    if let Some(object) = value.as_object_mut() {
        object.remove("saved_at");
    }
    serde_json::to_vec(&value)
        .map_err(|error| format!("Failed to normalize snapshot '{}': {error}", path.display()))
}

fn hash_component(hasher: &mut Sha256, name: &[u8], data: &[u8]) {
    hasher.update((name.len() as u64).to_le_bytes());
    hasher.update(name);
    hasher.update((data.len() as u64).to_le_bytes());
    hasher.update(data);
}

fn package_payload_offset(manifest: &ProjectPackageManifest) -> Result<u64, String> {
    let encoded = serde_json::to_vec(manifest)
        .map_err(|error| format!("Failed to encode package manifest: {error}"))?;
    Ok(PACKAGE_MAGIC.len() as u64 + 8 + encoded.len() as u64)
}

fn total_payload_bytes(manifest: &ProjectPackageManifest) -> Result<u64, String> {
    manifest.files.iter().try_fold(0_u64, |total, entry| {
        total
            .checked_add(entry.size_bytes)
            .ok_or_else(|| "Package payload size overflow".to_string())
    })
}

fn package_temp_path(output_path: &Path) -> PathBuf {
    let name = output_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("project.cuemap");
    output_path.with_file_name(format!(".{name}.{}.tmp", Uuid::new_v4()))
}

fn validate_id(project_id: &str) -> Result<(), String> {
    if validate_project_id(project_id) {
        Ok(())
    } else {
        Err(format!(
            "Invalid project ID '{project_id}'; use 3-64 letters, numbers, '-' or '_'"
        ))
    }
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
    use crate::engine::CueMapEngine;
    use crate::structures::MainStats;
    use tempfile::TempDir;

    fn create_snapshot(data_dir: &Path, project_id: &str, content: &str) {
        let snapshots = data_dir.join("snapshots");
        fs::create_dir_all(&snapshots).unwrap();
        let engine = CueMapEngine::<MainStats>::new();
        engine.add_memory(
            content.to_string(),
            vec!["portable".to_string()],
            None,
            MainStats::default(),
            true,
        );
        PersistenceManager::save_to_path(
            &engine,
            &snapshots.join(format!("{project_id}.bin")),
        )
        .unwrap();
    }

    #[test]
    fn package_round_trip_preserves_snapshot() {
        let source = TempDir::new().unwrap();
        let target = TempDir::new().unwrap();
        let package_dir = TempDir::new().unwrap();
        let package = package_dir.path().join("tiny.cuemap");
        create_snapshot(source.path(), "tiny_project", "portable memory");
        let contents = source.path().join("contents/tiny_project");
        let artifacts = source.path().join("artifacts/tiny_project/nested");
        fs::create_dir_all(&contents).unwrap();
        fs::create_dir_all(&artifacts).unwrap();
        fs::write(contents.join("sidecar.bin"), b"disk-backed content").unwrap();
        fs::write(artifacts.join("bridge.json"), b"{\"edge\":1}").unwrap();

        let packed = pack_project(source.path(), "tiny_project", &package, false).unwrap();
        assert_eq!(packed.file_count, 3);
        let manifest = inspect_project_package(&package).unwrap();
        assert_eq!(manifest.project_id, "tiny_project");

        let loaded = load_project_package(target.path(), &package, false).unwrap();
        assert_eq!(loaded.project_id, "tiny_project");
        let (memories, _, _, _, _) = PersistenceManager::load_from_path::<MainStats>(
            &target.path().join("snapshots/tiny_project.bin"),
        )
        .unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(
            memories.iter().next().unwrap().value().access_content(None).unwrap(),
            "portable memory"
        );
        assert_eq!(
            fs::read(target.path().join("contents/tiny_project/sidecar.bin")).unwrap(),
            b"disk-backed content"
        );
        assert_eq!(
            fs::read(target.path().join("artifacts/tiny_project/nested/bridge.json")).unwrap(),
            b"{\"edge\":1}"
        );
    }

    #[test]
    fn package_detects_payload_corruption() {
        let source = TempDir::new().unwrap();
        let target = TempDir::new().unwrap();
        let package_dir = TempDir::new().unwrap();
        let package = package_dir.path().join("tiny.cuemap");
        create_snapshot(source.path(), "tiny_project", "portable memory");
        pack_project(source.path(), "tiny_project", &package, false).unwrap();

        let mut bytes = fs::read(&package).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        fs::write(&package, bytes).unwrap();
        let error = load_project_package(target.path(), &package, false).unwrap_err();
        assert!(error.contains("Checksum mismatch"));
    }

    #[test]
    fn package_refuses_overwrite_without_force() {
        let source = TempDir::new().unwrap();
        let target = TempDir::new().unwrap();
        let package_dir = TempDir::new().unwrap();
        let package = package_dir.path().join("tiny.cuemap");
        create_snapshot(source.path(), "tiny_project", "new memory");
        create_snapshot(target.path(), "tiny_project", "old memory");
        pack_project(source.path(), "tiny_project", &package, false).unwrap();

        let error = load_project_package(target.path(), &package, false).unwrap_err();
        assert!(error.contains("already exists"));
        load_project_package(target.path(), &package, true).unwrap();
        let (memories, _, _, _, _) = PersistenceManager::load_from_path::<MainStats>(
            &target.path().join("snapshots/tiny_project.bin"),
        )
        .unwrap();
        assert_eq!(
            memories.iter().next().unwrap().value().access_content(None).unwrap(),
            "new memory"
        );
    }

    #[test]
    fn sync_state_hash_is_checked_before_package_installation() {
        let source = TempDir::new().unwrap();
        let accepted = TempDir::new().unwrap();
        let rejected = TempDir::new().unwrap();
        let package_dir = TempDir::new().unwrap();
        let package = package_dir.path().join("tiny.cuemap");
        create_snapshot(source.path(), "tiny_project", "new memory");
        create_snapshot(rejected.path(), "tiny_project", "old memory");
        pack_project(source.path(), "tiny_project", &package, false).unwrap();

        let expected = project_state_sha256(source.path(), "tiny_project").unwrap();
        load_project_package_with_state_hash(
            accepted.path(),
            &package,
            false,
            &expected,
        )
        .unwrap();

        let error = load_project_package_with_state_hash(
            rejected.path(),
            &package,
            true,
            &"f".repeat(64),
        )
        .unwrap_err();
        assert!(error.contains("project state hash mismatch"));
        let (memories, _, _, _, _) = PersistenceManager::load_from_path::<MainStats>(
            &rejected.path().join("snapshots/tiny_project.bin"),
        )
        .unwrap();
        assert_eq!(
            memories.iter().next().unwrap().value().access_content(None).unwrap(),
            "old memory"
        );
    }

    #[test]
    fn manifest_rejects_unsafe_paths() {
        let manifest = ProjectPackageManifest {
            format: PACKAGE_FORMAT.to_string(),
            version: PACKAGE_VERSION,
            engine_version: "test".to_string(),
            project_id: "safe_project".to_string(),
            created_at: 0,
            files: vec![ProjectPackageFile {
                path: "contents/../../escape".to_string(),
                size_bytes: 0,
                sha256: "0".repeat(64),
            }],
        };
        assert!(validate_manifest(&manifest).is_err());
    }

    #[test]
    fn s3_uris_resolve_without_accepting_non_s3_sources() {
        assert_eq!(
            s3_destination("s3://example-bucket/team/", "demo-project").unwrap(),
            "s3://example-bucket/team/demo-project.cuemap"
        );
        assert_eq!(
            s3_destination("s3://example-bucket", "demo-project").unwrap(),
            "s3://example-bucket/demo-project.cuemap"
        );
        assert!(validate_s3_uri("https://example.com/file", false).is_err());
        assert!(validate_s3_uri("s3://example-bucket", false).is_err());
    }

    #[test]
    fn project_state_hash_ignores_snapshot_save_time_but_tracks_content() {
        let source = TempDir::new().unwrap();
        create_snapshot(source.path(), "hash_project", "first memory");
        let before = project_state_sha256(source.path(), "hash_project").unwrap();

        let snapshot = source.path().join("snapshots/hash_project.bin");
        let raw = fs::read(&snapshot).unwrap();
        let decoded = zstd::stream::decode_all(io::Cursor::new(raw)).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&decoded).unwrap();
        value["saved_at"] = serde_json::json!(9_999_999_999_u64);
        let encoded = serde_json::to_vec(&value).unwrap();
        fs::write(
            &snapshot,
            zstd::stream::encode_all(io::Cursor::new(encoded), 3).unwrap(),
        )
        .unwrap();
        assert_eq!(
            project_state_sha256(source.path(), "hash_project").unwrap(),
            before
        );

        create_snapshot(source.path(), "hash_project", "different memory");
        assert_ne!(
            project_state_sha256(source.path(), "hash_project").unwrap(),
            before
        );
    }
}
