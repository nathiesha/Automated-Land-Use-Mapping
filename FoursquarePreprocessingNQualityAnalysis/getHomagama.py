import csv

counter = 0
records = 0
with open("original.csv", "rb") as inp, open("homagama.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if (float(row[1]) < 6.861113) and (float(row[1]) > 6.827366) and (float(row[2]) < 80.024242) and (float(row[2]) > 79.988365):
           writer.writerow(row)
           records += 1
        else:
            counter += 1
    print records

