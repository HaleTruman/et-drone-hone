function stableHash(value) {
  let hash = 2166136261;
  for (const character of String(value)) {
    hash ^= character.codePointAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function provenanceColor(value) {
  const hue = stableHash(value) % 360;
  const isBlue = hue >= 200 && hue < 270;
  const isPurple = hue >= 270 && hue < 330;
  const saturation = isBlue ? 90 : (isPurple ? 82 : 72);
  const lightness = isBlue ? 68 : (isPurple ? 70 : 58);
  return `hsl(${hue} ${saturation}% ${lightness}%)`;
}

export function applySchemaProvenance(element, provenance = {}) {
  const { fieldPath, fieldValue, evidencePath } = provenance;
  if (!fieldPath || fieldValue === undefined || fieldValue === null) return;
  const value = String(fieldValue);
  element.dataset.provenanceField = fieldPath;
  element.dataset.provenanceValue = value;
  if (evidencePath) element.dataset.evidenceField = evidencePath;
  element.style.setProperty('--provenance-color', provenanceColor(value));
  element.title = [
    `${fieldPath}=${value}`,
    evidencePath,
  ].filter(Boolean).join(' · ');
}
