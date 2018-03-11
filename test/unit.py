
print("Setting distances...")
distances = {}
#distances = []
length = len(shape)
for i in range(length):
    pfrom = shape[i]
    distances[i] = {}
    for j in range(i + 1, length):
        pto = shape[j]
        dis = ((pto[0] - pfrom[0]) ** 2 + (pto[1] - pfrom[1]) ** 2) ** 0.5
        #distances.append((i, j, dis))
        distances[i][j] = dis

pp.pprint(distances)
ratios = []


for p1_idx, vals1 in distances.items():
    for p2_idx, dis in vals1.items():
        for p3_idx, vals2 in distances.items():
            ratio = 0
            if p3_idx != p2_idx and p3_idx != p1_idx:
                if p3_idx < p2_idx:
                    ratio = dis / vals2[p2_idx]
                elif p3_idx == p2_idx:
                    ratios.extend([
                        (p1_idx, p2_idx, i, dis / v) for i, v in vals2.items()])
            if ratio != 0:
                ratios.append((p1_idx, p2_idx, p3_idx, ratio))
#pp.pprint(ratios) 
total = 0
for (p1_idx, p2_idx, p3_idx, r) in ratios:
    for score, error in error_levels.items():
        if abs(r - golden_ratio) < score:
            total += score
            rgb = tuple(random.randint(0, 255) for _ in range(3))

            p1 = tuple(shape[p1_idx])
            p2 = tuple(shape[p2_idx])
            p3 = tuple(shape[p3_idx])
            
            """
            cv2.line(image, p1, p2, rgb)
            cv2.line(image, p2, p3, rgb)
            """
print("TOTAL: " + str(total))
