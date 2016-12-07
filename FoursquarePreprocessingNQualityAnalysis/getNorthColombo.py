import csv

counter = 0
records = 0
with open("original.csv", "rb") as inp, open("dehiwala.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if (float(row[1]) < 6.863616) and (float(row[1]) > 6.809842) and (float(row[2]) < 79.893326) and (float(row[2]) > 79.862599):
           writer.writerow(row)
           records += 1
        else:
            counter += 1
    print counter
    print records

