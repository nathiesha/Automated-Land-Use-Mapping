import csv

counter = 0
records = 0
with open("original.csv", "rb") as inp, open("borellaC.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if (float(row[1]) < 6.924275) and (float(row[1]) > 6.899459) and (float(row[2]) < 79.888902) and (float(row[2]) > 79.875426):
           writer.writerow(row)
           records += 1
        else:
            counter += 1
    print records

