import argparse
from google_images_download import google_images_download



parser = argparse.ArgumentParser(description='fill arguement')

parser.add_argument('--find', type=str, required=True,
                    help='the word for searching, use , to split ')

parser.add_argument('--min', required=False, type=int, default=2,
                    help="the threshold of match count")

parser.add_argument('--url', required=False, default="https://www.google.com.tw/imghp?hl=zh-TW&authuser=0&ogbl",
                    help="the iamge's url")


args = parser.parse_args()


txt = args.find


response = google_images_download.googleimagesdownload()
arguments = {"keywords": txt,"limit": args.min,"print_urls":True}
absolute_image_paths = response.download(arguments)
print("images paths", absolute_image_paths)


