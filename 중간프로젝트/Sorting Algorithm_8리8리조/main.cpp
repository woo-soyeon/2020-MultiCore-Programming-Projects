#include <stdio.h>
#include <stdlib.h>
#include <omp.h> 
#include "DS_definitions.h"
#include "DS_timer.h"

void merge(int* a, int* b, int left, int mid, int right);
void merge_sort(int* a, int* b, int left, int right);
void swap(int* xp, int* yp);
void bubble_sort(int* a, int left, int right);
void heapify(int* a, int here, int size);
void buildHeap(int* a, int size);
void serial_heap_sort(int* a, int size);
void parallel_heap_sort(int *data, int num);

int main(int argc, char *argv[]) {
	DS_timer timer(6);

	timer.setTimerName(0, (char*)"Merge Sort(Serial)");
	timer.setTimerName(1, (char*)"Merge Sort(Parallel)");
	timer.setTimerName(2, (char*)"Bubble Sort(Serial)");
	timer.setTimerName(3, (char*)"Bubble Sort(Parallel)");
	timer.setTimerName(4, (char*)"Heap Sort(Serial)");
	timer.setTimerName(5, (char*)"Heap Sort(Parallel)");

	int n = atoi(argv[1]); // �迭�� ũ��
	int p = omp_get_max_threads(); // �������� ����
	int m = n / p;

	const int SIZE = n * sizeof(int);

	int* a = (int*)malloc(sizeof(int)*SIZE);
	int* b = (int*)malloc(sizeof(int)*SIZE);

	int* serial_result = (int*)malloc(sizeof(int)*SIZE);
	int* parallel_result = (int*)malloc(sizeof(int)*SIZE);

	printf("Size of Array : %d", n);
	printf("\n");
	printf("\n");

	int i;

	// �迭�� 1~500 ������ �ֱ�
	for (i = 0; i < n; i++) {
		a[i] = rand() % 500 + 1;
	}

	// �迭 ����Ʈ
	/*printf("Print Array\n");

	for (i = 0; i < n; i++) {
		printf("%d ", a[i]);
	}
	printf("\n");
	printf("\n");*/

	printf("********** MERGE SORT **********\n");

	// Serial Merge Sort
	timer.onTimer(0);

	merge_sort(a, b, 0, n - 1);

	serial_result[i] = a[i];

	timer.offTimer(0);

	/*printf("Print Serial Merge Sort\n");
	for (i = 0; i < n; i++) {
		printf("%d ", a[i]);
		serial_result[i] = a[i];
	}
	printf("\n");
	printf("\n");*/

	// Parallel Merge Sort
	timer.onTimer(1);

	#pragma omp parallel 
	{
		int tid = omp_get_thread_num();
		int start = m * tid;
		int end = m * (tid + 1);

		if (tid < p - 1) // ������ ������ ���� ��
			merge_sort(a, b, start, end);
		else // ������ ������
			merge_sort(a, b, start, n - 1);
	}

	for (i = 1; i < p - 1; i++) { // ������ ������ ���� ��
		int middle = i * m;
		merge(a, b, 0, middle, middle + m - 1);
	}

	merge(a, b, 0, (p - 1) * m, n - 1); // ������ ������

	parallel_result[i] = a[i];

	timer.offTimer(1);

	/*printf("print parallel merge sort\n");
	for (i = 0; i < n; i++) {
		printf("%d ", a[i]);
		parallel_result[i] = a[i];
	}
	printf("\n");
	printf("\n");*/

	// Serial����� Parallel��� ��
	bool isCorrect = true;

	for (int i = 0; i < n; i++) {
		if (serial_result[i] != parallel_result[i]) {
			isCorrect = true;
			//printf("%d\n", serial_result[i]);
			//printf("%d\n", parallel_result[i]);
			break;
		}
	}
	if (isCorrect)
		printf("Results are matched! :)\n");
	else
		printf("Results are not matched :(\n");

	printf("\n");
	*serial_result = 0;
	*parallel_result = 0;

	printf("********** BUBBLE SORT **********\n");

	// Serial Bubble Sort
	timer.onTimer(2);

	bubble_sort(a, 0, n - 1);

	serial_result[i] = a[i];

	timer.offTimer(2);

	/*printf("Print Serial Bubble Sort\n");
	for (i = 0; i < n; i++) {
		printf("%d ", a[i]);
		serial_result[i] = a[i];
	}
	printf("\n");
	printf("\n");*/

	// Parallel Bubble Sort
	timer.onTimer(3);
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int start = m * tid;
		int end = m * (tid + 1);

		if (tid < p - 1) // ������ ������ ���� ��
			bubble_sort(a, start, end);
		else // ������ ������
			bubble_sort(a, start, n - 1);
	}

	for (i = 1; i < p - 2; i++) { // ������ ������ ���� ��
		int middle = i * m;
		merge(a, b, 0, middle, middle + m - 1);
	}

	merge(a, b, 0, (p - 1) * m, n - 1); // ������ ������

	parallel_result[i] = a[i];

	timer.offTimer(3);

	/*printf("Print Parallel Bubble Sort\n");
	for (i = 0; i < n; i++) {
		printf("%d ", a[i]);
		parallel_result[i] = a[i];
	}
	printf("\n");
	printf("\n");*/

	// Serial����� Parallel��� ��
	bool isCorrect_2 = true;

	for (int i = 0; i < n; i++) {
		if (serial_result[i] != parallel_result[i]) {
			isCorrect_2 = true;
			break;
		}
	}

	if (isCorrect_2)
		printf("Results are matched! :)\n");
	else
		printf("Results are not matched :(\n");

	printf("\n");
	*serial_result = 0;
	*parallel_result = 0;

	printf("********** HEAP SORT **********\n");

	// Serial Heap Sort
	timer.onTimer(4);

	serial_heap_sort(a, n);

	serial_result[i] = a[i];

	timer.offTimer(4);

	/*printf("Print Serial Heap Sort\n");
	for (i = 0; i < n; i++) {
		printf("%d ", a[i]);
		serial_result[i] = a[i];
	}
	printf("\n");
	printf("\n");*/

	// Parallel Heap Sort
	timer.onTimer(5);

	parallel_heap_sort(a, n);

	parallel_result[i] = a[i];

	timer.offTimer(5);


	/*printf("Print Parallel Heap Sort\n");
	for (i = 0; i < n; i++) {
		printf("%d ", a[i]);
		parallel_result[i] = a[i];
	}
	printf("\n");
	printf("\n");*/

	// Serial����� Parallel��� ��
	bool isCorrect_3 = true;

	for (int i = 0; i < n; i++) {
		if (serial_result[i] != parallel_result[i]) {
			isCorrect_3 = true;
			break;
		}
	}

	if (isCorrect_3)
		printf("Results are matched! :)\n");
	else
		printf("Results are not matched :(\n");

	printf("\n");

	timer.printTimer();
	EXIT_WIHT_KEYPRESS;
}

// �պ� ����
void merge_sort(int* a, int* b, int left, int right) {
	int mid;
	if (right > left) {
		mid = (right + left) / 2;
		merge_sort(a, b, left, mid); // �պκ� ����Ʈ ����
		merge_sort(a, b, mid + 1, right); // �޺κ� ����Ʈ ����
		merge(a, b, left, mid + 1, right); // ���ĵ� 2���� �迭 �պ�
	}
}

// 2���� ���ĵ� �迭 �պ�
void merge(int* a, int* b, int left, int mid, int right) {
	int i, left_end, count;
	int j;
	left_end = mid - 1;
	j = left;
	count = right - left + 1;

	// ���� ���ĵ� �迭 �պ�
	while ((left <= left_end) && (mid <= right)) {
		if (a[left] <= a[mid]) {
			b[j] = a[left];
			j++;
			left++;
		}
		else {
			b[j] = a[mid];
			j++;
			mid++;
		}
	}

	// ���ʿ� ���� �ִ� �� ����
	while (left <= left_end) {
		b[j] = a[left];
		left++;
		j++;
	}

	// �����ʿ� ���� �ִ� �� ����
	while (mid <= right) {
		b[j] = a[mid];
		mid++;
		j++;
	}

	// ���� �迭�� �纹��
	for (i = 0; i < count; i++) {
		a[right] = b[right];
		right--;
	}
}

void swap(int* xp, int* yp) {
	int temp = *xp;
	*xp = *yp;
	*yp = temp;
}

// ���� ����
void bubble_sort(int* a, int left, int right) {
	int i, j;
	for (i = left; i < right; i++)
		for (j = left; j < right - i; j++)
			if (a[j] > a[j + 1])
				swap(&a[j], &a[j + 1]);
}

// ũ�Ⱑ size�� �迭 a�� here���� �� ������� ����� �Լ�
void heapify(int* a, int here, int size) {
	int left = here * 2 + 1;
	int right = here * 2 + 2;
	int max = here;
	if (left < size && a[left]>a[max])
		max = left;
	if (right < size && a[right]>a[max])
		max = right;

	if (max != here) {
		swap(&a[here], &a[max]);
		heapify(a, max, size);
	}
}

// �� �����
void buildHeap(int* a, int size) {
	int i;
	for (i = size  - 1; i >= 0; i--) { // size / 2 - 1 = �θ����� �ε���
		heapify(a, i, size);
	}
}

// ���� �� ����
void serial_heap_sort(int* a, int size) {
	int index;
	buildHeap(a, size);
	for (index = size - 1; index >= 0; index--) {
		swap(&a[0], &a[index]);
		heapify(a, 0, index);
	}
}

// ���� �� ����
void parallel_heap_sort(int* a, int size) {
	int index;
	buildHeap(a, size);

	#pragma omp parallel for private(index) shared(a, size) num_threads(1)
	for (index = size - 1; index >= 0; index--) {
		swap(&a[0], &a[index]);
		heapify(a, 0, index);
	}
}

void print_array(int* a, int n) {
	for (int i = 0; i < n; i++) {
		printf("%d ", a[i]);
	}
	printf("\n");
	printf("\n");
}
