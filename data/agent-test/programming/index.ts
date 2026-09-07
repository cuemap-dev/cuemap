export function index(values: string[]): Map<string, number> {
  return new Map(values.map((value, index) => [value, index]));
}
