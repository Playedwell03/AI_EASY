// Does the data need regenerating? Compare against the settings the current data was built from,
// not against the previous settings — presets and "back to defaults" replace hp wholesale.
export function dataNeedsRebuild(dataKeys, builtFrom, hp) {
  if (!dataKeys) return true;
  return dataKeys.some((k) => builtFrom[k] !== hp[k]);
}
