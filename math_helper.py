import matplotlib.pyplot as plt

def display_data(self):
		for DS in self.targetDS:
			U = self.outputs[DS][1]
			V = self.outputs[DS][2]
			#plot news sources
			y = U[1]
			x = U[2]
			n = self.dataArray[DS][3]
			fig, ax = plt.subplots()
			ax.scatter(z, y)
			for i, txt in enumerate(n):
				ax.annotate(txt, (z[i], y[i]))
			plt.title("Topic %d", DS)