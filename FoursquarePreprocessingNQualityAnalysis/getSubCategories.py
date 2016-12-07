import csv

sub =set()
counter = 0
with open("original.csv", "rb") as inp, open("Subcategories.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[3] not in sub:
            writer.writerow(row)
            sub.add(row[3])
    print sub.__len__()