import numpy as np
import matplotlib.pyplot as plt


def make_1D_rules(number: int):
    """
    Takes a number and returns np-array with binary values
    :param number:
    :return:
    """
    assert number > -1
    assert number < 256
    rule_array = np.zeros([8])
    binary_string = "{0:b}".format(number)
    nValues = len(binary_string)
    for i in range(nValues):
        i += 1
        rule_array[-i] = int(binary_string[-i])

    return rule_array


def get_rule(number: int, rule_idx: int):
    """
    Finds what the updated state should be based on rule nr. and decimal value of pattern above.
    :param number:
    :param rule_idx:
    :return:
    """
    rules = make_1D_rules(number)
    rule_idx += 1
    rule_idx = int(rule_idx)
    return rules[-rule_idx]


def binatodeci(binary):
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))


def update_cell(number, input):
    # Assuming input is a list with three elements
    """
    Gives value of cell based on rule number and list of above cell values
    :param number:
    :param input:
    :return:
    """
    rule_idx = binatodeci(input)
    return int(get_rule(number, rule_idx))


# TODO ska man göra objekt för celler och uppdatera states på dessa eller bara pilla med listor typ
# Nu kan man skicka in en siffra och lista med 3 element och få ut nytt värde för cellen


width = 100
parent_generation = np.random.randint(0,2, width, int)
# parent_generation = np.zeros(width)
# parent_generation[25] = 1
nGenerations = 100
rule_nr = 145

print("RULE:")
print(make_1D_rules(rule_nr))

fig, ax = plt.subplots()


for row in range(nGenerations):
    if row == 0:
        matrix = parent_generation
        new_row = 0.5*np.ones([nGenerations-1, width])
        matrix = np.vstack([matrix, new_row])
        ax.imshow(matrix)

        plt.pause(5)
    else:

        for col in range(width):
            if col == width-1:
                values_above = [matrix[row-1, col-1], matrix[row-1, col], matrix[row-1, 0]]
            else:
                values_above = [matrix[row-1, col-1], matrix[row-1, col], matrix[row-1, col+1]]

            matrix[row, col] = update_cell(rule_nr, values_above)
    plt.gca().clear()
    ax.imshow(matrix)
    plt.pause(0.001)
# plt.imshow(matrix)
plt.show()



# plt.savefig('outputs/rule{}'.format(rule_nr))
# print(matrix)