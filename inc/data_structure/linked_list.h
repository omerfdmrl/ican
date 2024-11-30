#ifndef ICAN_DATASTRUCTURE_LINKEDLIST

#define ICAN_DATASTRUCTURE_LINKEDLIST

#include "ican.h"

struct s_single_linked_list {
    int data;
    struct s_single_linked_list *next;
};

typedef struct s_single_linked_list SingleLinkedList;

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

#endif // !ICAN_DATASTRUCTURE_LINKEDLIST