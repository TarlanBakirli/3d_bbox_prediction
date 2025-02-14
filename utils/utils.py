import numpy as np
import cv2
import torch
import math

from scipy.spatial.transform import Rotation

def get_oriented_bboxes(masks):

    num_objects = masks.shape[0]
    bboxes = []
    bboxes_csa = []
    
    for i in range(num_objects):
        mask = masks[i]
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        
        rect = cv2.minAreaRect(contour)
        
        center = rect[0]
        size = rect[1]
        angle = rect[2]
        
        vertices = cv2.boxPoints(rect)
        
        bbox_info = {
            'center': center,       # (x,y) tuple
            'size': size,           # (width,height) tuple
            'angle': angle,         # rotation angle in degrees
            'vertices': vertices    # 4 corners as numpy array
        }

        bboxes.append(vertices)
        bboxes_csa.append(np.array([center[0], center[1], size[0], size[1], math.radians(angle)]))
        
    return np.array(bboxes), np.array(bboxes_csa)

def is_points_in_bbox_vectorized(points_2d, bbox):
    vertices = bbox
    
    inside_mask = np.ones(len(points_2d), dtype=bool)
    
    for i in range(4):
        # Get current edge vertices
        v1 = vertices[i]
        v2 = vertices[(i + 1) % 4]
        
        # Edge vector
        edge = v2 - v1
        
        # Normal vector (perpendicular to edge, pointing inward)
        normal = np.array([-edge[1], edge[0]])
        
        # Vector from vertex to points
        to_points = points_2d - v1
        
        # Dot product with normal determines which side the point is on
        side = np.dot(to_points, normal)
        
        # Update mask - point must be on the correct side of all edges
        inside_mask &= (side >= 0)
    
    return inside_mask

def filter_pointcloud_by_bboxes_vectorized(pointcloud, bboxes, debug=False):
    H, W = pointcloud.shape[1:]
    
    # Create arrays of pixel coordinates
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    points_2d = np.stack([x_coords, y_coords], axis=-1).reshape(-1, 2)
    
    # Initialize mask for all points
    points_mask = np.zeros(H*W, dtype=bool)
    
    # Check points against all bboxes
    for bbox in bboxes:
        bbox_mask = is_points_in_bbox_vectorized(points_2d, bbox)
        points_mask |= bbox_mask
    
    # Reshape mask back to image dimensions
    mask = points_mask.reshape(H, W)
    
    # Create filtered pointcloud
    filtered_pointcloud = pointcloud.copy()
    filtered_pointcloud[:, ~mask] = np.nan
    
    return filtered_pointcloud, mask

def convert_pc_to_standard_format(filtered_pointcloud):
    points = filtered_pointcloud.reshape(3, -1)
    points = points.T
    valid_mask = ~np.isnan(points).any(axis=1)
    points = points[valid_mask]
    
    return points