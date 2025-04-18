# Параллельное вычисление суммы элементов массива MPI
Программа демонстрирует параллельное суммирование элементов массива с использованием технологии MPI (Message Passing Interface).

## Особенности реализации
-Распределение данных между процессами с помощью MPI_Scatter

-Параллельное вычисление частичных сумм на каждом процессе

-Сбор результатов с использованием MPI_Reduce

-Сравнение производительности последовательной и параллельной версий

-Генерация тестовых данных с использованием <random>
### Входные параметры

| Параметр         | Описание                                   | Пример значения         |
|------------------|--------------------------------------------|-------------------------|
| Размер массива   | Количество элементов в массиве             | 1,000,000 элементов     |
| Число процессов  | Количество MPI-процессов (задается через `-n`) | 4                      |

### Выходные данные

| Метод        | Формат вывода                     | Пример вывода               |
|--------------|-----------------------------------|-----------------------------|
| Sequential   | `[сумма] (Time: [время] sec)`     | 504823 (Time: 0.125 sec)    |
| MPI_Reduce   | `[сумма] (Time: [время] sec)`     | 504823 (Time: 0.042 sec)    |
| Speedup      | `[послед. время]/[паралл. время]` | 2.98x                      |

**Пояснения:**
- `Speedup` рассчитывается как отношение времени последовательной версии к параллельной
- Размер массива можно изменить в исходном коде программы
- Количество процессов задается при запуске: `mpiexec -n 4 ./program`
