# import os
# import multiprocessing as mp
# from imutils import resize
# import pymp
# import cv2
#
# import setup
# from core import faces, landmarks, all_distances, ratios, feed_ratio,\
#     show_img, extract_rect, calc_avg_ratios, calc_ratio_stds,
#     compare_with_avg
# from utils import show_img_with_lands, visible_only
# from config import IMG_PATH, RATIOS_PATH, DATASET_PATH, PROCESSES
# from helpers import pp, sl
#
#
# # Use this to calculate standard deviations and write it to file
# """
# calc_ratio_stds()
# """
#
# # Use this to calculate averages and write it to file
# """
# calc_avg_ratios()
# """
#
# # Use this to feed ratios.json
# """
# img_names = os.listdir(DATASET_PATH)
# _feed_ratios(img_names)
# """
#
# # Use this for multiprocessing
# """
# img_names = os.listdir(DATASET_PATH)
# proc_list = []
#
# imgs_per_proc = len(img_names) / PROCESSES
# for i in range(PROCESSES):
#     from_idx = int(i * imgs_per_proc)
#     to_idx = int((i + 1) * imgs_per_proc)
#     pp(from_idx)
#     pp(to_idx)
#
#     imgs_for_proc = img_names[from_idx:to_idx]
#     proc_list.append(mp.Process(target=_feed_ratios, args=(imgs_for_proc,)))
#
# for process in proc_list:
#     process.start()
#     process.join()
# """
#
# """ --- FUNCTIONS --- """
#
#
# def _feed_ratio(img_name):
#     """ Assumes one face per image """
#     img = cv2.imread(DATASET_PATH + img_name)
#     #img = resize(img, width=1000)
#
#     dlib_face = faces(img)[0]
#     # Open this to convert all faces to same width
#     """
#     face_only = extract_rect(img, face)
#     face_resized = resize(face_only, width=500)
#
#     dlib_face = faces(face_resized)[0]
#     """
#     lands = landmarks(img, dlib_face)
#
#     all_dists = all_distances(lands, allow_repeats=True)
#     rs = ratios(all_dists)
#     feed_ratio(rs)
#
#
# def _feed_ratios(img_names):
#     for img_name in img_names:
#         _feed_ratio(img_name)


# Use this to calculate lengths
#     """
#     with open(RATIOS_PATH, 'r') as r:
#         for line in r:
#             pp(len(json.loads(line)))
#     """

# img_names = (
#         'me-frontal.jpg',
#         'hd-girl.jpg',
#         'hd-boy.jpg',
#         'me.jpg',
#         'us.jpg',
#         'mona-lisa.jpg')
