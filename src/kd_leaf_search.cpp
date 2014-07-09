
#include "kd_fix_rad_search.h"			// kd fixed-radius search decls

//----------------------------------------------------------------------
//	Approximate  k nearest neighbor Leaf search
//              The idea here is to merely find the leaf which covers the query point of interest
//              and return all elements in that leaf. The utility is in doing an *extremely* fast
//              search for nearest neighbor, using a tree with large bucket sizes. This way, the
//              buckets themselves act as filters for data. The distances within the bucket are not
//              calculated.
//
//		Note: This was derived from the method in annkFRSearch, so it has some cruft left
//		over from this function. All it is supposed to do is return all the elements in a
//              bucket. There's probably a more clever way to do this with a direct pointer, but
//              the perfect is the enemy of the "pretty OK."
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//		To keep argument lists short, a number of global variables
//		are maintained which are common to all the recursive calls.
//		These are given below.
//----------------------------------------------------------------------

int				ANNkdLeafDim;				// dimension of space
ANNpoint		ANNkdLeafQ;				// query point
double			ANNkdLeafMaxErr;			// max tolerable squared error
ANNpointArray	ANNkdLeafPts;				// the points
ANNmin_k*		ANNkdLeafPointMK;			// set of k closest points
int				ANNkdLeafPtsVisited;		// total points visited
int				ANNkdLeafPtsInRange;		// number of points in the range

//----------------------------------------------------------------------
//	annkLeafSearch - fixed radius search for k nearest neighbors
//----------------------------------------------------------------------

int ANNkd_tree::annkLeafSearch(
	ANNpoint			q,				// the query point
	int					k,				// number of near neighbors to return
	ANNidxArray			nn_idx,			// nearest neighbor indices (returned)
	double				eps)			// the error bound
{

	ANNkdLeafDim = dim;					// copy arguments to static equivs
	ANNkdLeafQ = q;
	ANNkdLeafPts = pts;
	ANNkdLeafPtsVisited = 0;				// initialize count of points visited
	ANNkdLeafPtsInRange = 0;				// ...and points in the range

	ANNkdLeafMaxErr = ANN_POW(1.0 + eps);
	ANN_FLOP(2)							// increment floating op count

	ANNkdLeafPointMK = new ANNmin_k(k);	// create set for closest k points
										// search starting at the root
	root->ann_Leaf_search(annBoxDistance(q, bnd_box_lo, bnd_box_hi, dim));

	for (int i = 0; i < k; i++) {		// extract the k-th closest points
		if (nn_idx != NULL)
			nn_idx[i] = ANNkdLeafPointMK->ith_smallest_info(i);
	}

	delete ANNkdLeafPointMK;				// deallocate closest point set
	return ANNkdLeafPtsInRange;			// return final point count
}

//----------------------------------------------------------------------
//	kd_split::ann_Leaf_search - search a splitting node
//		Note: This routine is similar in structure to the standard kNN
//		search.  It visits the subtree that is closer to the query point
//		first.
//----------------------------------------------------------------------

void ANNkd_split::ann_Leaf_search(ANNdist box_dist)
{
										// check dist calc term condition
	if (ANNmaxPtsVisited != 0 && ANNkdLeafPtsVisited > ANNmaxPtsVisited) return;

										// distance to cutting plane
	ANNcoord cut_diff = ANNkdLeafQ[cut_dim] - cut_val;

	if (cut_diff < 0) {					// left of cutting plane
		child[ANN_LO]->ann_Leaf_search(box_dist);// visit closer child first

		ANNcoord box_diff = cd_bnds[ANN_LO] - ANNkdLeafQ[cut_dim];
		if (box_diff < 0)				// within bounds - ignore
			box_diff = 0;
										// distance to further box
		box_dist = (ANNdist) ANN_SUM(box_dist,
				ANN_DIFF(ANN_POW(box_diff), ANN_POW(cut_diff)));

		if (box_dist * ANNkdLeafMaxErr < ANNkdLeafPointMK->max_key())
		  child[ANN_HI]->ann_Leaf_search(box_dist);

	}  else {								// right of cutting plane
		child[ANN_HI]->ann_Leaf_search(box_dist);// visit closer child first

		ANNcoord box_diff = ANNkdLeafQ[cut_dim] - cd_bnds[ANN_HI];
		if (box_diff < 0)				// within bounds - ignore
			box_diff = 0;
										// distance to further box
		box_dist = (ANNdist) ANN_SUM(box_dist,
				ANN_DIFF(ANN_POW(box_diff), ANN_POW(cut_diff)));

		if (box_dist * ANNkdLeafMaxErr < ANNkdLeafPointMK->max_key())
		  child[ANN_LO]->ann_Leaf_search(box_dist);

	}
	ANN_FLOP(13)						// increment floating ops
	ANN_SPL(1)							// one more splitting node visited
}

//----------------------------------------------------------------------
//	kd_leaf::ann_Leaf_search - search points in a leaf node
//		Note: This was derived from the method in annkFRSearch, so it has some cruft left
//		over from this function. All it is supposed to do is return all the elements in a
//              bucket. There's probably a more clever way to do this with a direct pointer, but
//              the perfect is the enemy of the "pretty OK."
//----------------------------------------------------------------------

void ANNkd_leaf::ann_Leaf_search(ANNdist box_dist)
{
	register ANNdist dist;				// distance to data point
	register ANNcoord* qq;				// query coordinate pointer
	register ANNcoord t;
	register int d;

	for (int i = 0; i < n_pts; i++) {	// check points in bucket
												// add it to the list
			ANNkdLeafPointMK->insert(dist, bkt[i]);
			ANNkdLeafPtsInRange++;				// increment point count

	}
	ANN_LEAF(1)							// one more leaf node visited
	ANN_PTS(n_pts)						// increment points visited
	ANNkdLeafPtsVisited += n_pts;			// increment number of points visited
}
