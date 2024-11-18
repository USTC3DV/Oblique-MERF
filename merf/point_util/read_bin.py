# 作者 任晨曲
import collections
import os
import struct
import numpy as np
import open3d as o3d

Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def points3d_denoise(points3D, outPath):
    points = []
    colors = []
    for point in points3D.values():
        points.append(point.xyz)
        colors.append(point.rgb)

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 显示原始点云
    # o3d.visualization.draw_geometries_with_animation_callback([pcd], custom_draw_geometry_with_animation_callback)
    # 创建统计滤波器
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3., print_progress=True)
    # 根据滤波器结果剔除离群点
    pcd_filtered = pcd.select_by_index(ind)
    # 显示去噪后的点云
    # o3d.visualization.draw_geometries_with_animation_callback([pcd_filtered], custom_draw_geometry_with_animation_callback)
    # 将点云保存为PLY文件
    o3d.io.write_point_cloud(outPath, pcd_filtered, print_progress=True)


def read_bin_output_ply(in_path,out_path):
    # 读取点云
    points3D = read_points3d_binary(in_path)
    # 点云去噪
    points3d_denoise(points3D, out_path)

if __name__ == '__main__':
    # 点云.bin文件所在目录位置
    path = "/home/zxy/LandMark/data/ustc/colmap/sparse/1"
    # 读取点云
    points3D = read_points3d_binary(os.path.join(path, 'points3D.bin'))
    # 点云去噪
    points3d_denoise(points3D, os.path.join(path, 'output_test2.ply'))
