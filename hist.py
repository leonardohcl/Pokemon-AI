from Pokemon import Pokedex, PokemonType
from matplotlib import pyplot as plt

colors = {
    PokemonType.Grass:  "#78c850",
    PokemonType.Fire:  "#F08030",
    PokemonType.Water:  "#6890F0",
    PokemonType.Bug:  "#A8B820",
    PokemonType.Normal:  "#A8A878",
    PokemonType.Poison:  "#A040A0",
    PokemonType.Electric:  "#F8D030",
    PokemonType.Ground:  "#E0C068",
    PokemonType.Fairy:  "#EE99AC",
    PokemonType.Fighting:  "#C03028",
    PokemonType.Psychic:  "#F85888",
    PokemonType.Rock:  "#B8A035",
    PokemonType.Ghost:  "#705898",
    PokemonType.Ice:  "#78c850",
    PokemonType.Dragon:  "#98D8D8",
    PokemonType.Flying:  "#66c3e8",
    PokemonType.Dark:  "#472e07",
    PokemonType.Steel:  "#828282",
}

dex = Pokedex("pokemon.csv")
types = [PokemonType(i) for i in range(18)]

counts = [len(dex.get_type(t)) for t in types]
print(counts)
plt.bar([str(t) for t in types], counts, color=[colors[t] for t in types])
plt.show()
