import csv

# read the data
data = []
with open('./outputs/anomalies.csv', 'r') as f:
    data = list(csv.reader(f))

# filter out the first 100 data points, since they are just noise
data = data[100:]

# sort the data
data.sort(key=lambda x: float(x[0]), reverse=True)

# output the anomalies
with open('./outputs/anomalies_sorted.csv', 'w') as f:
    for row in data:
        f.write(','.join(row) + '\n')
