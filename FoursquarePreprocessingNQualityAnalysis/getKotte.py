import csv

counter = 0
records = 0
with open("original.csv", "rb") as inp, open("rathmalana.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if (float(row[1]) < 6.883993) and (float(row[1]) > 6.866610) and (float(row[2]) < 79.929086) and (float(row[2]) > 79.901706):
           writer.writerow(row)
           records += 1
        else:
            counter += 1
    print records

