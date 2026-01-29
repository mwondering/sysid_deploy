import os
import trimesh

# 根目录（当前目录）
root_dir = "."

# 遍历所有子目录中的 .obj 文件
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(".obj"):
            obj_path = os.path.join(subdir, file)
            stl_path = os.path.splitext(obj_path)[0] + ".stl"

            try:
                mesh = trimesh.load(obj_path, force='mesh')
                mesh.export(stl_path)
                print(f"✅ Converted: {obj_path} -> {stl_path}")
            except Exception as e:
                print(f"❌ Failed: {obj_path}\n   Reason: {e}")
