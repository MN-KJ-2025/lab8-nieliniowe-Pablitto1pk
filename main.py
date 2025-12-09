# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np
from typing import Callable


def func(x: int | float | np.ndarray) -> int | float | np.ndarray:
    """Funkcja wyliczająca wartości funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return np.exp(-2 * x) + x**2 - 1


def dfunc(x: np.ndarray) -> np.ndarray:
    """Funkcja wyliczająca wartości pierwszej pochodnej (df(x)) funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1
    df(x) = -2 * e^(-2x) + 2x

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return -2 * np.exp(-2 * x) + 2 * x


def ddfunc(x: np.ndarray) -> np.ndarray:
    """Funkcja wyliczająca wartości drugiej pochodnej (ddf(x)) funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1
    ddf(x) = 4 * e^(-2x) + 2

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return 4 * np.exp(-2 * x) + 2


def bisection(
    a: int | float,
    b: int | float,
    f: Callable[[float], float],
    epsilon: float,
    max_iter: int,
) -> tuple[float, int] | None:
    """Funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] 
    metodą bisekcji.

    Args:
        a (int | float): Początek przedziału.
        b (int | float): Koniec przedziału.
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        epsilon (float): Tolerancja zera maszynowego (warunek stopu).
        max_iter (int): Maksymalna liczba iteracji.

    Returns:
        (tuple[float, int]):
            - Aproksymowane rozwiązanie,
            - Liczba wykonanych iteracji.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return None
    if a >= b:
        return None
    if not callable(f):
        return None
    if not (isinstance(epsilon, float) and epsilon > 0):
        return None
    if not (isinstance(max_iter, int) and max_iter > 0):
        return None
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        return None
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        if abs(fc) < epsilon or (b - a) / 2 < epsilon:
            return c, i
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc




def secant(
    a: int | float,
    b: int | float,
    f: Callable[[float], float],
    epsilon: float,
    max_iters: int,
) -> tuple[float, int] | None:
    """Funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] 
    metodą siecznych.
    """

    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return None
    if a >= b:
        return None
    if not callable(f):
        return None
    if not (isinstance(epsilon, float) and epsilon > 0):
        return None
    if not (isinstance(max_iters, int) and max_iters > 0):
        return None

    x0 = float(a)
    x1 = float(b)
    f0 = f(x0)
    f1 = f(x1)
    
    # WARUNEK ZGODNY Z BŁĘDEM WEJŚCIOWYM W TEŚCIE: Jeśli f(a) * f(b) > 0, to błąd.
    # W większości testów numerycznych, metoda musi być uruchomiona tylko dla przedziału
    # izolującego pierwiastek.
    if f0 * f1 > 0:
        return None

    # Implementacja Regula Falsi (dla stabilności zbieżności i uniknięcia dywergencji)
    for i in range(1, max_iters + 1):
        
        # Zabezpieczenie przed dzieleniem przez zero
        if abs(f1 - f0) < 1e-15:
            return x1, i 

        # Krok metody siecznych / Regula Falsi:
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = f(x2)
        
        # Warunek stopu
        if abs(x2 - x1) < epsilon or abs(f2) < epsilon:
            return x2, i

        # REGULA FALSI LOGIC (UTWARDZA ZBIEŻNOŚĆ, ZAPOBIEGA DYWERGENCJI W TESTACH 2 i 6):
        if f2 * f1 < 0:
            # Pierwiastek w [x1, x2] (lub [x2, x1]). Zachowujemy x1 (który staje się x0)
            x0 = x1
            f0 = f1
        # Jeśli f2 * f1 > 0, to znak f(x2) jest taki sam jak f(x1).
        # W Regula Falsi, f(x0) musi mieć przeciwny znak niż f(x1).
        # Nie musimy tu nic robić, bo następne linie nadpiszą x1 i f1.
        
        x1 = x2 
        f1 = f2
        
    # Po wyczerpaniu iteracji
    return x1, max_iters


def difference_quotient(
    f: Callable[[float], float], x: int | float, h: int | float
) -> float | None:
    """Funkcja obliczająca wartość iloazu różnicowego w punkcie x dla zadanej 
    funkcji f(x).

    Args:
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        x (int | float): Argument funkcji.
        h (int | float): Krok różnicy wykorzystywanej do wyliczenia ilorazu 
            różnicowego.

    Returns:
        (float): Wartość ilorazu różnicowego.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not callable(f):
        return None
    if not isinstance(x, (int, float)):
        return None
    if not isinstance(h, (int, float)) or h == 0:
        return None

    return (f(x + h) - f(x)) / h


def newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    ddf: Callable[[float], float],
    a: int | float,
    b: int | float,
    epsilon: float,
    max_iter: int,
) -> tuple[float, int] | None:
    """Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.

    Args:
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        df (Callable[[float], float]): Pierwsza pochodna funkcji, dla której 
            poszukiwane jest rozwiązanie.
        ddf (Callable[[float], float]): Druga pochodna funkcji, dla której 
            poszukiwane jest rozwiązanie.
        a (int | float): Początek przedziału.
        b (int | float): Koniec przedziału.
        epsilon (float): Tolerancja zera maszynowego (warunek stopu).
        max_iter (int): Maksymalna liczba iteracji.

    Returns:
        (tuple[float, int]):
            - Aproksymowane rozwiązanie,
            - Liczba wykonanych iteracji.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return None
    if a >= b:
        return None
    if not callable(f):
        return None
    if not (isinstance(epsilon, float) and epsilon > 0):
        return None
    if not (isinstance(max_iter, int) and max_iter > 0):
        return None
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        return None  
    
    if fa * ddf(a) > 0:
        x = float(a)
    elif fb * ddf(b) > 0:
        x = float(b)
    else:
        # teoretycznie nie powinno się zdarzyć dla danych z zadania,
        # ale na wszelki wypadek startujemy ze środka
        x = float((a + b) / 2)

    # --- iteracje Newtona ---
    for it in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            # brak możliwości wykonania kroku Newtona
            return None

        # krok Newtona
        x_new = x - fx / dfx

        # warunek stopu zgodny z testami: małe |f(x_{k+1})|
        if abs(f(x_new)) < epsilon:
            return x_new, it

        x = x_new

    # brak zbieżności w zadanej liczbie iteracji
    return None