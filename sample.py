import matplotlib.pyplot as plt

fig = plt.gcf()
fig.set_size_inches(16, 5)

title_list = ['Train Loss', 'Test Acc', 'Test Loss']

value_list = [
	[
		
	],
	[
	],
	[
	],

]

legend_list = [
	['benchmark', '2 workers', '4 workers', '5 workers', '8 workers', '10 workers', '16 workers'],
	['benchmark', '2 workers', '4 workers', '5 workers', '8 workers', '10 workers', '16 workers'],
	['benchmark', '2 workers', '4 workers', '5 workers', '8 workers', '10 workers', '16 workers'],
]

for i in range(3):
	plt.subplot(1, 3, i+1)
	for value in value_list[i]:
		plt.plot(value)
	plt.title(title_list[i])
	plt.xlabel('epoch')
	plt.ylabel(title_list[i])
	plt.legend(legend_list[i], loc='best')

plt.savefig('avg2-16.png')



