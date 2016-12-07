import csv

counter = 0
records = 0
with open("original.csv", "rb") as inp, open("moratuwa.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if (float(row[1]) < 6.813012) and (float(row[1]) > 6.762471) and (float(row[2]) < 79.903200) and (float(row[2]) > 79.875048):
           writer.writerow(row)
           records += 1
        else:
            counter += 1
    print records

