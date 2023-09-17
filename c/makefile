progs = naive_matmul cache_matmul thread_matmul
src = $(progs:=.c)
headers = stopwatch.h

CC = clang
FLAGS = -Wall -Wextra -Werror 

all: $(progs) 
clean:
	rm -f $(progs) *.o


%: %.c $(headers)
	$(CC) $(FLAGS) -o $@ $<