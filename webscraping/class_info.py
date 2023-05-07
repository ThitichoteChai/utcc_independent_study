class Comment_info:
    def __init__(self):
        self.product_name = None
        self.url_name = None
        self.overall_rate = None
        self.date = None
        self.cat = None
        self.num_like = None
        self.review_rate = None
        self.comment = None
    
    def assign_product_name(self, take):
        self.product_name = take

    def get_product_name(self):
        return self.product_name
        
    ### ------------------------------------------------------------------------
        
    def assign_url_name(self, take):
        self.url_name = take

    def get_url_name(self):
        return self.url_name
    
    ### ------------------------------------------------------------------------

    def assign_cat(self, take):
        self.cat = take

    def get_cat(self):
        return self.cat
    
    ### ------------------------------------------------------------------------
    
    def assign_overall_rate(self, take):
        self.overall_rate = take

    def get_overall_rate(self):
        return self.overall_rate
    
    ### ------------------------------------------------------------------------
    
    def assign_num_like(self, take):
        self.num_like = take

    def get_num_like(self):
        return self.num_like
    
    ### ------------------------------------------------------------------------
    
    def assign_review_rate(self, take):
        self.review_rate = take

    def get_review_rate(self):
        return self.review_rate
    
    ### ------------------------------------------------------------------------
    
    def assign_comment(self, take):
        self.comment = take

    def get_comment(self):
        return self.comment