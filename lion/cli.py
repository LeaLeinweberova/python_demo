import lion

dataset = lion.load_dataset()

vysledek_split = lion.split_dataset(dataset)

lion.test_models(vysledek_split[0], vysledek_split[2])

lion.prediction_model(vysledek_split[0], vysledek_split[2], vysledek_split[1], vysledek_split[3])

"""

#lion.print_dataset(dataset)
#lion.plot_dataset(dataset, x=False, y=False)
#lion.plot_dataset(dataset, mykind='hist', x=False, y=False)

"""