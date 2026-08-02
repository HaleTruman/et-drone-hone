export const PNP_WORLD_COLORS = Object.freeze({
  rawCurrent: 0x52e8ff,
  secondaryCurrent: 0xffb347,
  finalCurrent: 0xff4fd8,
  rawHistory: 0xffd84a,
  finalHistory: 0x55f29a,
  elapsedTrajectory: 0xe6edf3,
  fullTrajectory: 0x64748b,
  drone: 0xffd84a,
  camera: 0xffffff,
  depthHalo: 0x020305,
});

export const PNP_WORLD_SCENE_CONFIGURATION = Object.freeze({
  cameraFovDeg: 55,
  cameraNearM: 0.05,
  cameraFarM: 5000,
  gridSizeM: 120,
  gridDivisions: 60,
  currentRawBorderPx: 8,
  currentSecondaryBorderPx: 3,
  currentFinalBorderPx: 4,
  historyRawBorderPx: 3,
  historyFinalBorderPx: 2,
  depthBorderPx: 2,
  droneAxisLengthM: 1.25,
  cameraAxisLengthM: 0.8,
});
