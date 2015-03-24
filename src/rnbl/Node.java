package rnbl;


import java.util.*;
import weka.core.Instances;


public class Node {
	Node parent;
	LinkedList<Node> children;
	Instances D;
	
	public Node() {
		children = new LinkedList<Node>();
	}
}
