import lion
import sys
 
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("nasol 4 hodnoty")
        sys.exit(1)

    sepallength = float(sys.argv[1])
    sepalwidth = float(sys.argv[2])
    petallength = float(sys.argv[3])
    petalwidth = float(sys.argv[4])
    iris_class = lion.predict_iris_class(sepallength, sepalwidth, petallength, petalwidth)
    print('Class ',iris_class)