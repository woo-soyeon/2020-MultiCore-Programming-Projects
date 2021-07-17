#include <stdio.h>
#include <stdlib.h>
#include <omp.h> 
#include "DS_definitions.h"
#include "DS_timer.h"

void merge(int* a, int* b, int left, int mid, int right);
void merge_sort(int* a, int* b, int left, int right);

int main(int argc, char *argv[]) {
	DS_timer timer(2);

	timer.setTimerName(0, (char*)"Merge Sort(Serial)");
	timer.setTimerName(1, (char*)"Merge Sort(Parallel)");
	
	int n = atoi(argv[1]); // �迭�� ũ��
	int p = omp_get_num_threads(); // �������� ����
	int m = n / p;

	const int SIZE = n * sizeof(int); 

	int* a = (int*)malloc(SIZE);
	int* b = (int*)malloc(SIZE);

	int* serial_result = (int*)malloc(SIZE);
	int* parallel_result = (int*)malloc(SIZE);

	printf("Size of Array : %d", n);
	printf("\n");
	printf("\n");

	int i;
	
	// �迭�� 1~500 ������ �ֱ�
	for (i = 0; i < n; i++) {
		a[i] = rand() % 500 + 1;
	}
	
	// �迭 ����Ʈ
	printf("Print Array\n");

	for (i = 0; i < n; i++) {
		printf("%d ", a[i]);
	}
	printf("\n");
	printf("\n");

	// Serial Merge Sort
	timer.onTimer(0);

	merge_sort(a, b, 0, n-1);

	printf("Print Serial Merge Sort\n");
	for (i = 0; i < n; i++) {
		printf("%d ", a[i]);
		serial_result[i] = a[i];
	}
	printf("\n");
	printf("\n");

	timer.offTimer(0);

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

	printf("Print Parallel Merge Sort\n");
	for (i = 0; i < n; i++) {
		printf("%d ", a[i]);
		parallel_result[i] = a[i];
	}
	printf("\n");
	printf("\n");

	timer.offTimer(1);

	// Serial����� Parallel��� ��
	bool isCorrect = true;

	for (int i = 0; i < n; i++) {
		if (serial_result[i] != parallel_result[i]) {
			isCorrect = false;
			break;
		}
	}

	if (isCorrect)
		printf("Results are matched! :)\n");
	else
		printf("Results are not matched :(\n");

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
		// for (i = 0; i <= count; i++) {
		a[right] = b[right];
		right--;
	}
}
