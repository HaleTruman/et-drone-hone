import math

import cv2

from .schema import (CORNER_ROLES, CornerGeometry, ParentContourCorner,
                     VoidDetectionFeatures)


def parent_obtuse_corners(contour, minimum_degrees=160.0):
    points = contour.reshape(-1, 2)
    corners = []
    for index, point in enumerate(points):
        left = points[index - 1] - point
        right = points[(index + 1) % len(points)] - point
        scale = math.hypot(*left) * math.hypot(*right)
        if not scale:
            continue
        cosine = max(-1.0, min(1.0, float(left @ right) / scale))
        angle = math.degrees(math.acos(cosine))
        if angle > minimum_degrees:
            corners.append(ParentContourCorner(
                tuple(map(int, point)), round(angle, 3)))
    return tuple(corners)


def inner_corners(contours, center, axes):
    cx, cy = center
    enclosing = [contour for contour in contours
                 if cv2.pointPolygonTest(contour, center, False) >= 0]
    if not enclosing:
        return ()
    target_area = 3.14159 * axes[0] * axes[1]
    boundary = min(enclosing, key=lambda contour:
                   abs(cv2.contourArea(contour) - target_area)).reshape(-1, 2)
    limit = (axes[0] + axes[1]) ** 2
    eligible = [point for point in boundary
                if (point[0] - cx) ** 2 + (point[1] - cy) ** 2 <= limit]
    chosen = []
    for role, (dx, dy) in zip(
            CORNER_ROLES, ((-1, -1), (1, -1), (-1, 1), (1, 1))):
        used = {item[1] for item in chosen}
        pool = [point for point in eligible if tuple(point) not in used]
        if pool:
            point = max(pool, key=lambda value:
                        (value[0] - cx) * dx + (value[1] - cy) * dy)
            chosen.append((role, tuple(map(int, point))))
    return tuple(CornerGeometry(
        f"inner_{role}", point, "void_contour") for role, point in chosen)


def compute_geometry(contours, groups):
    features = []
    for parent, center, axes, _ in groups:
        inner_points = inner_corners(contours, center, axes)
        outer_points, chosen = [], set()
        cx, cy = center
        boundary = contours[parent].reshape(-1, 2)
        for role, (dx, dy) in zip(
                CORNER_ROLES, ((-1, -1), (1, -1), (-1, 1), (1, 1))):
            available = (value for value in boundary
                         if tuple(map(int, value)) not in chosen)
            edge = max(available, key=lambda value:
                       (int(value[0]) - cx) * dx + (int(value[1]) - cy) * dy)
            edge = tuple(map(int, edge))
            chosen.add(edge)
            vx, vy = edge[0] - cx, edge[1] - cy
            length = max(1, math.hypot(vx, vy))
            radius = min(length, 1.15 * sum(axes))
            point = (round(cx + radius * vx / length),
                     round(cy + radius * vy / length))
            outer_points.append(CornerGeometry(
                f"outer_{role}", point, "parent_contour"))
        features.append(VoidDetectionFeatures(
            tuple(outer_points), tuple(inner_points),
            parent_obtuse_corners(contours[parent])))
    return features


def draw_outer_corners(image, features, colors):
    for feature, color in zip(features, colors):
        for corner in feature.outer_corners:
            x, y = corner.point_px
            cv2.rectangle(image, (x - 4, y - 4), (x + 4, y + 4),
                          (0, 0, 0), cv2.FILLED)
            cv2.rectangle(image, (x - 3, y - 3), (x + 3, y + 3),
                          color, cv2.FILLED)


def draw_inner_corners(image, features, colors):
    for feature, color in zip(features, colors):
        for corner in feature.inner_corners:
            cv2.drawMarker(
                image, corner.point_px, color, cv2.MARKER_TILTED_CROSS, 7, 2)


def draw_parent_obtuse_corners(image, features):
    corners = {corner.point_px: corner for feature in features
               for corner in feature.parent_obtuse_corners}
    for corner in corners.values():
        cv2.drawMarker(
            image, corner.point_px, (0, 255, 255), cv2.MARKER_DIAMOND, 9, 2)


def draw_geometry(image, features, colors):
    draw_outer_corners(image, features, colors)
    draw_inner_corners(image, features, colors)
    draw_parent_obtuse_corners(image, features)


def render_geometry(image, contours, groups):
    features = compute_geometry(contours, groups)
    draw_geometry(image, features, [group[3] for group in groups])
    return features
