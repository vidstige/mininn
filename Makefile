SRC := .
OBJ := .

SOURCES := $(wildcard $(SRC)/*.c)
OBJECTS := $(patsubst $(SRC)/%.c, $(OBJ)/%.o, $(SOURCES))

scnn: $(OBJECTS)
	$(CC) $^ -o $@ -lm

$(OBJ)/%.o: $(SRC)/%.c
	$(CC) -Wall -I$(SRC) -c $< -o $@

clean:
	rm $(OBJECTS) scnn

