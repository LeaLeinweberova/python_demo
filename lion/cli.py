import lion

dataset = lion.load_dataset()

#lion.print_dataset(dataset)
#lion.plot_dataset(dataset, x=False, y=False)
#lion.plot_dataset(dataset, mykind='hist', x=False, y=False)

lion.test_dataset(dataset)