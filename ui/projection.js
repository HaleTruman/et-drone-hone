(function exposeProjection(root) {
  'use strict';

  // Camera right/down/forward; image-edge coordinates, native centers at i + .5.
  // The illustrated depth follows the active model, not the conflicting PDF diagram.
  const constants = Object.freeze({
    width: 640, height: 360, fx: 320, fy: 320, cx: 320, cy: 180,
    pixelCenterOffset: 0.5, outer: 2.7, inner: 1.5, depth: 0.26
  });
  const defaultPose = Object.freeze({x: 0, y: 0, z: 6, roll: 0, pitch: 0, yaw: 25});
  const controls = Object.freeze([
    {key: 'x', label: 'X · right', min: -3, max: 3, step: 0.1, unit: 'm', axis: 'X'},
    {key: 'y', label: 'Y · down', min: -3, max: 3, step: 0.1, unit: 'm', axis: 'Y'},
    {key: 'z', label: 'Z · forward', min: 3, max: 18, step: 0.1, unit: 'm', axis: 'Z'},
    {key: 'roll', label: 'Roll · Z', min: -180, max: 180, step: 1, unit: '°', axis: 'Z'},
    {key: 'pitch', label: 'Pitch · X', min: -70, max: 70, step: 1, unit: '°', axis: 'X'},
    {key: 'yaw', label: 'Yaw · Y', min: -70, max: 70, step: 1, unit: '°', axis: 'Y'}
  ].map(Object.freeze));
  const o = constants.outer / 2, i = constants.inner / 2, d = constants.depth / 2;
  const rails = [
    [[-o, -o, -d], [-i, o, d]], [[i, -o, -d], [o, o, d]],
    [[-i, -o, -d], [i, -i, d]], [[-i, i, -d], [i, o, d]]
  ];
  const faceIndices = [[0,1,3,2], [4,6,7,5], [0,4,5,1], [2,3,7,6], [0,2,6,4], [1,5,7,3]];
  const colors = ['#2a6856', '#438773', '#34765f', '#2d6e58', '#4f927b', '#397e66'];
  const epsilon = 1e-9;

  function checkedPose(pose) {
    const value = {...defaultPose, ...pose};
    for (const control of controls) {
      if (!Number.isFinite(value[control.key]) || value[control.key] < control.min || value[control.key] > control.max) {
        throw new RangeError('Invalid illustrative pose: ' + control.key);
      }
    }
    return value;
  }

  function hull(points) {
    const sorted = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (a, b, c) => (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]);
    const lower = [], upper = [];
    for (const point of sorted) {
      while (lower.length > 1 && cross(lower[lower.length-2], lower[lower.length-1], point) <= 0) lower.pop();
      lower.push(point);
    }
    for (let n = sorted.length - 1; n >= 0; n--) {
      const point = sorted[n];
      while (upper.length > 1 && cross(upper[upper.length-2], upper[upper.length-1], point) <= 0) upper.pop();
      upper.push(point);
    }
    return lower.slice(0, -1).concat(upper.slice(0, -1));
  }

  function pixelCount(hulls, bounds) {
    // Each cuboid projects to a convex polygon. Merge its inclusive integer spans
    // per native row, so overlapping rails count once and the opening stays empty.
    const offset = constants.pixelCenterOffset;
    const first = Math.max(0, Math.ceil(bounds.minY - offset - epsilon));
    const last = Math.min(constants.height - 1, Math.floor(bounds.maxY - offset + epsilon));
    let count = 0;
    for (let row = first; row <= last; row++) {
      const y = row + offset, intervals = [];
      for (const polygon of hulls) {
        const intersections = [];
        for (let n = 0; n < polygon.length; n++) {
          const a = polygon[n], b = polygon[(n + 1) % polygon.length];
          if (y < Math.min(a[1], b[1]) - epsilon || y > Math.max(a[1], b[1]) + epsilon) continue;
          if (Math.abs(b[1] - a[1]) < epsilon) {
            if (Math.abs(y - a[1]) < epsilon) intersections.push(a[0], b[0]);
          } else {
            intersections.push(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
          }
        }
        if (!intersections.length) continue;
        const left = Math.max(0, Math.ceil(Math.min(...intersections) - offset - epsilon));
        const right = Math.min(constants.width - 1, Math.floor(Math.max(...intersections) - offset + epsilon));
        if (right >= left) intervals.push([left, right]);
      }
      intervals.sort((a, b) => a[0] - b[0]);
      let end = -1;
      for (const [left, right] of intervals) {
        count += Math.max(0, right - Math.max(left, end + 1) + 1);
        end = Math.max(end, right);
      }
    }
    return count;
  }

  function projectGeometry(pose) {
    const state = checkedPose(pose);
    const radians = Math.PI / 180;
    const cr = Math.cos(state.roll*radians), sr = Math.sin(state.roll*radians);
    const cp = Math.cos(state.pitch*radians), sp = Math.sin(state.pitch*radians);
    const cy = Math.cos(state.yaw*radians), sy = Math.sin(state.yaw*radians);
    // Apply Rz(roll), then Rx(pitch), then Ry(yaw), all about the gate center.
    const transform = point => {
      const x = cr*point[0] - sr*point[1], y = sr*point[0] + cr*point[1];
      const py = cp*y - sp*point[2], pz = sp*y + cp*point[2];
      return [cy*x + sy*pz + state.x, py + state.y, -sy*x + cy*pz + state.z];
    };
    const uv = point => [constants.cx + constants.fx*point[0]/point[2], constants.cy + constants.fy*point[1]/point[2]];
    const faces = [], hulls = [], all = [];
    for (const [lo, hi] of rails) {
      const vertices = [];
      for (let x = 0; x < 2; x++) for (let y = 0; y < 2; y++) for (let z = 0; z < 2; z++) {
        vertices.push(transform([lo[0]+x*(hi[0]-lo[0]), lo[1]+y*(hi[1]-lo[1]), lo[2]+z*(hi[2]-lo[2])]));
      }
      // z >= 3 m exceeds the gate's bounding-sphere radius for every allowed RPY.
      const points = vertices.map(uv);
      all.push(...points);
      hulls.push(hull(points));
      faceIndices.forEach((indices, n) => faces.push({
        points: indices.map(index => points[index]),
        z: indices.reduce((sum, index) => sum + vertices[index][2], 0) / indices.length,
        color: colors[n]
      }));
    }
    faces.sort((a, b) => b.z - a.z);
    return {
      faces, hulls, center: uv([state.x, state.y, state.z]),
      bounds: {
        minX: Math.min(...all.map(point => point[0])), minY: Math.min(...all.map(point => point[1])),
        maxX: Math.max(...all.map(point => point[0])), maxY: Math.max(...all.map(point => point[1]))
      }
    };
  }

  function project(pose = defaultPose) {
    const result = projectGeometry(pose);
    result.pixelCount = pixelCount(result.hulls, result.bounds);
    return result;
  }

  function randomPose(random = Math.random) {
    const margin = 12;
    for (let attempt = 0; attempt < 100; attempt++) {
      const pose = {};
      for (const control of controls) {
        const sample = random();
        if (!Number.isFinite(sample) || sample < 0 || sample >= 1) throw new RangeError('Random source must return values in [0, 1).');
        const steps = Math.round((control.max - control.min) / control.step);
        pose[control.key] = Number((control.min + Math.floor(sample*(steps + 1))*control.step).toFixed(1));
      }
      const bounds = projectGeometry(pose).bounds;
      if (bounds.minX >= margin && bounds.minY >= margin && bounds.maxX <= constants.width - margin && bounds.maxY <= constants.height - margin) return pose;
    }
    return {...defaultPose};
  }

  const api = Object.freeze({constants, controls, defaultPose, project, randomPose});
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.GuideProjection = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
