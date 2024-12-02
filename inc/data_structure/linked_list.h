#ifndef ICAN_DATASTRUCTURE_LINKEDLIST_H

#define ICAN_DATASTRUCTURE_LINKEDLIST_H

#include "ican.h"

struct s_single_linked_list {
    int data;
    struct s_single_linked_list *next;
};
struct s_double_linked_list {
    int data;
    struct s_double_linked_list *next;
    struct s_double_linked_list *prev;
};

typedef struct s_single_linked_list SingleLinkedList;
typedef struct s_double_linked_list DoubleLinkedList;

SingleLinkedList *single_linked_list_alloc(int data);
void single_linked_list_free(SingleLinkedList *head);
SingleLinkedList *single_linked_list_insert_beginning(SingleLinkedList *head, int data);
SingleLinkedList *single_linked_list_insert_end(SingleLinkedList *head, int data);
SingleLinkedList *single_linked_list_insert_pos(SingleLinkedList *head, int data, int16 pos);
SingleLinkedList *single_linked_list_delete_beginning(SingleLinkedList *head);
SingleLinkedList *single_linked_list_delete_end(SingleLinkedList *head);
SingleLinkedList *single_linked_list_delete_pos(SingleLinkedList *head, int16 pos);
SingleLinkedList *single_linked_list_search(SingleLinkedList *head, int target) ;
void single_linked_list_traverse(SingleLinkedList *head);

DoubleLinkedList *double_linked_list_alloc(int data);
void double_linked_list_free(DoubleLinkedList *head);
DoubleLinkedList *double_linked_list_insert_beginning(DoubleLinkedList *head, int data);
DoubleLinkedList *double_linked_list_insert_end(DoubleLinkedList *head, int data);
DoubleLinkedList *double_linked_list_insert_pos(DoubleLinkedList *head, int data, int16 pos);
DoubleLinkedList *double_linked_list_search(DoubleLinkedList *head, int target);
void double_linked_list_traverse_forward(DoubleLinkedList *head);
void double_linked_list_traverse_backward(DoubleLinkedList *head);

#endif // !ICAN_DATASTRUCTURE_LINKEDLIST_H