import cv2

from .schema import (CORNER_ROLES, CircleGeometry, CornerGeometry,
                     VoidDetectionFeatures)


def radial_circle(role, center, corner):
    point = corner.point_px
    return CircleGeometry(
        role, center, round(cv2.norm(center, point)), (corner.role,), (point,))


def diameter_circle(outer, inner):
    outer_point, inner_point = outer.point_px, inner.point_px
    center = (round((outer_point[0] + inner_point[0]) / 2),
              round((outer_point[1] + inner_point[1]) / 2))
    return CircleGeometry(
        "outer_inner_connection", center,
        round(cv2.norm(outer_point, inner_point) / 2),
        (outer.role, inner.role), (outer_point, inner_point))


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


def render_geometry(image, contours, groups):
    projected, inner, features = [], [], []
    for parent, center, axes, color in groups:
        inner_points = inner_corners(contours, center, axes)
        inner.extend((center, corner, color) for corner in inner_points)
        outer_points, pairs = [], []
        cx, cy = center
        boundary = contours[parent].reshape(-1, 2)
        chosen = set()
        for role, (dx, dy) in zip(
                CORNER_ROLES, ((-1, -1), (1, -1), (-1, 1), (1, 1))):
            available = (value for value in boundary
                         if tuple(map(int, value)) not in chosen)
            edge = max(available, key=lambda value:
                       (int(value[0]) - cx) * dx + (int(value[1]) - cy) * dy)
            edge = tuple(map(int, edge))
            chosen.add(edge)
            vx, vy = edge[0] - cx, edge[1] - cy
            length = max(1, (vx * vx + vy * vy) ** 0.5)
            radius = min(length, 1.15 * sum(axes))
            point = (round(cx + radius * vx / length),
                     round(cy + radius * vy / length))
            corner = CornerGeometry(
                f"outer_{role}", point, "parent_contour")
            projected.append((center, corner, color))
            outer_points.append(corner)
        for corner in outer_points:
            if inner_points:
                farthest = max(inner_points, key=lambda inner_point:
                               cv2.norm(corner.point_px, inner_point.point_px))
                pairs.append((corner, farthest))
        features.append(VoidDetectionFeatures(
            tuple(outer_points), tuple(inner_points),
            tuple(radial_circle("ellipse_to_outer", center, corner)
                  for corner in outer_points),
            tuple(radial_circle("ellipse_to_inner", center, corner)
                  for corner in inner_points),
            tuple(diameter_circle(outer, inner_point)
                  for outer, inner_point in pairs)))
    for feature, group in zip(features, groups):
        color = group[3]
        circles = (feature.outer_corner_circles,
                   feature.inner_corner_circles, feature.connection_circles)
        for family in circles:
            for circle in family:
                cv2.circle(image, circle.center_px, circle.radius_px,
                           (0, 0, 0), 3)
                cv2.circle(image, circle.center_px, circle.radius_px, color, 1)
    for _, corner, color in projected:
        cv2.circle(image, corner.point_px, 4, (0, 0, 0), cv2.FILLED)
        cv2.circle(image, corner.point_px, 3, color, cv2.FILLED)
    for _, corner, color in inner:
        cv2.drawMarker(
            image, corner.point_px, color, cv2.MARKER_TILTED_CROSS, 7, 2)
    return features
