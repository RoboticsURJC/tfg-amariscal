import matplotlib.pyplot as plt

f = open('train.txt', 'r')
lines = f.readlines()

EPOCHS = 7
x = []
y = []

for line in lines:
    y.append(float(line))

f.close()

increment = float(EPOCHS / len(y))
count = increment
for i in range(0, len(y)):
    x.append(count)
    count += increment

plt.rcParams['toolbar'] = 'None'
figure = plt.figure(figsize=(100, 100))

figure.tight_layout()
ax = figure.add_subplot()
plt.axis([0, EPOCHS, 0, 0.3])
ax.set_ylabel('Loss', fontsize=15)
ax.set_xlabel('Epochs', fontsize=15)
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
ax.plot(x, y)

plt.show()
