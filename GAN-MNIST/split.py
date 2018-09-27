from PIL import Image

# I may have mixed the weight and height

work_num = 100	# 25, 50 or 100
x_n, y_n = 14, 14
h, w = 28, 28
label_n = 10
sample_n = 7	# each class sample number

load_path = './figures/sample_%04d.jpg' % work_num
save_path = './figures/sorted_%04d.jpg' % work_num

img = Image.open(load_path)

label = [
			[3, 1, 5, 6, 0, 5, 4, 5, 7, 6, 7, 2, 7, 9],
			[2, 3, 4, 3, 6, 9, 9, 9, 5, 3, 8, 6, 8, 0],
			[3, 2, 3,-1, 1, 9, 6, 1, 4, 9, 3, 1, 1, 3],
			[4, 4, 2, 7, 3, 1, 4, 9, 4, 9, 7, 4, 7, 8],
			[5, 8, 9, 5, 2, 5, 1, 5, 0, 7, 3, 6, 8, 8],
			[0, 1, 6, 0, 6, 4, 4, 1, 8, 5, 5, 9, 4, 6],
			[6, 7, 8, 1, 0, 3, 8, 9, 0, 4, 8, 9, 6, 4],
			[9, 5, 5, 8, 7, 2, 9, 7, 2, 1, 6, 1, 8, 2],
			[8, 2, 7, 7, 8, 8, 7, 3, 7, 7, 9, 5, 0, 9],
			[1, 4, 7, 7, 8, 3, 3, 6, 1,-1, 3,-1, 4, 3],
			[5, 9, 9, 0, 3, 5, 4, 0, 4, 3, 4, 1, 4, 7],
			[6, 5, 9, 3, 5, 9, 2, 0, 4, 0, 0, 7, 2, 1],
			[8, 3, 1, 8, 2, 0, 5, 5, 5, 2, 5, 9, 4, 0],
			[1, 6, 1, 0, 0, 5, 4, 9, 9, 7, 7, 6, 3, 4]
		]

# count = np.zeros(shape=[3, 10])

image_bucket = []
for i in range(label_n):
	image_bucket.append([])
	
# print(image_bucket)
# print(type(img))
# print(img.shape())


def cut_down(i, j):
	return img.crop((j*h, i*w, (j+1)*h, (i+1)*w))

# def piece_together():
# 	pass

# Classify
for i in range(x_n):
	for j in range(y_n):
		if label[i][j] != -1:
			image_bucket[label[i][j]].append(cut_down(i, j))

# piece_together
dest_im = Image.new('L', (sample_n * w, label_n * h))

for i in range(label_n):
	for j in range(sample_n):
		dest_im.paste(image_bucket[i][j], (j*h, i*w))


dest_im.save(save_path)

