#!/usr/bin/python3
import sys

lines = []
lines2 = []

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

with open(sys.argv[2], 'r') as f:
    lines2 = f.readlines()

average = [float(i) for i in lines[0].split()]
median = [float(i) for i in lines[1].split()]

average2 = [float(i) for i in lines2[0].split()]
median2 = [float(i) for i in lines2[1].split()]

print("AVERAGE:")

if float(average[0]) > float(average2[0]):
    print(f"{sys.argv[1]} is better in success rate ({average[0] - average2[0]}).")
elif float(average[0]) < float(average2[0]):
    print(f"{sys.argv[2]} is better in success rate ({average2[0] - average[0]}).")
else:
    print("Files have the same success rate.")

if float(average[1]) > float(average2[1]):
    print(f"{sys.argv[1]} is faster ({average[1] - average2[1]}).")
elif float(average[1]) < float(average2[1]):
    print(f"{sys.argv[2]} is faster ({average2[1] - average[1]}).")
else:
    print("Files are equally fast.")

if float(average[2]) > float(average2[2]):
    print(f"{sys.argv[1]} explores more states ({average[2] - average2[2]}).")
elif float(average[2]) < float(average2[2]):
    print(f"{sys.argv[2]} explores more states ({average2[2] - average[2]}).")
else:
    print("Files explore same number of states.")

print("MEDIAN:")

if float(median[0]) > float(median2[0]):
    print(f"{sys.argv[1]} is better in success rate ({median[0] - median2[0]}).")
elif float(median[0]) < float(median2[0]):
    print(f"{sys.argv[2]} is better in success rate ({median2[0] - median[0]}).")
else:
    print("Files have the same success rate.")

if float(median[1]) > float(median2[1]):
    print(f"{sys.argv[1]} is faster ({median[1] - median2[1]}).")
elif float(median[1]) < float(median2[1]):
    print(f"{sys.argv[2]} is faster ({median2[1] - median[1]}).")
else:
    print("Files are equally fast.")

if float(median[2]) > float(median2[2]):
    print(f"{sys.argv[1]} explores more states ({median[2] - median2[2]}).")
elif float(median[2]) < float(median2[2]):
    print(f"{sys.argv[2]} explores more states ({median2[2] - median[2]}).")
else:
    print("Files explore same number of states.")