import csv

cat =set()
counter = 0
with open("original.csv", "rb") as inp, open("categories.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[4] not in cat:
            writer.writerow(row)
            cat.add(row[4])
    print cat.__len__()