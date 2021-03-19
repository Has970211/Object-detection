class CreateD(object):
    def __init__(self, outputs):
       self.outputs=outputs

    def create(self):
        l=self.outputs['instances']
        dict1={}
        for i in range(len(l)):
            d={}
            a=l[i].get_fields()
            z=tuple(a['pred_boxes'])
            z=z[0].tolist()
            d['pred_box']=z
            
            x=a['scores']
            x=x.tolist()
            d['score']=x[0]

            y=a['pred_classes']
            y=y.tolist()
            d['pred_class']=y[0]
            
            dict1[str(i)]=d
         return dict1
