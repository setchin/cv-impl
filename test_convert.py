def convert_coord(x, y, z):
    # Gazebo: X North, Y West, Z Up
    # AOS goal: X East, Y South
    x_new = -y
    y_new = -x
    z_new = z
    return x_new, y_new, z_new

x, y, z = -40, -40, 70
print(f"Pose 0 converted: {convert_coord(x, y, z)}")

with open("photo_shoot_tuned/random_dem.obj", "r") as fin, open("converted_dem.obj", "w") as fout:
    for line in fin:
        if line.startswith("v "):
            parts = line.strip().split()
            x_v, y_v, z_v = float(parts[1]), float(parts[2]), float(parts[3])
            xn, yn, zn = convert_coord(x_v, y_v, z_v)
            fout.write(f"v {xn} {yn} {zn}\n")
        else:
            fout.write(line)

print("Created converted_dem.obj")
