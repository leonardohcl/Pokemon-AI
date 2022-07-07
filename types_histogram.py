from Pokemon import Pokedex, PokemonType
from matplotlib import pyplot as plt


dex = Pokedex("pokemon.csv")
types = [PokemonType(i) for i in range(18)]
counts = [len(dex.get_type(t)) for t in types]

plt.bar([str(t) for t in types], counts, color=[t.color() for t in types])
plt.show()
