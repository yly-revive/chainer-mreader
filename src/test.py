from argparse import ArgumentParser

if __name__ == '__main__':

	parser = ArgumentParser()

	parser.add_argument('--learning-rate', type=int, default=1)

	args = parser.parse_args()

	print(args.learning_rate)