name = "KLH"
print("{:<10}".format(name)) #left Alignment
print("{:>10}".format(name)) #left Alignment
print("{:^10}".format(name)) #Center Alignment
num = -250
print("{:=8}".format(num)) #Sign Alignment: Spaces are inserted between the sign and the number.
print("{:+}".format(45))
print("{:+}".format(-45)) #Always Show Sign
print("{:-}".format(-45))
num = 45.6789 
print("{:f}".format(num)) #Fixed-Point Format: automatically displays 6 digits after the decimal point.