SHELL	:=	/bin/bash

MODULES	:=	numpy pandas matplotlib typing_extensions

OUTPUTS	:=	logreg_coef.csv houses.csv sub

NAME	:=	venv

all		:	$(NAME)

$(NAME)	:
			python3 -m venv $(NAME)
			source ./$(NAME)/bin/activate; python3 -m pip install $(MODULES);

clean	:
			rm -rf $(OUTPUTS)

fclean	:	clean
			rm -rf $(NAME)

re		:	fclean all

.PHONY	:	all clean fclean re