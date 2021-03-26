class np_im(object):
  def __init__(self, outputs, PATH_im, filename):
    self.outputs=outputs
    self.PATH_im=PATH_im
    self.filename=filename
  
  def create(self):
    l=self.outputs['instances']
    for i in range(len(l)):
      a=l[i].get_fields()
      q=a['pred_masks']
      q=q.tolist()
          
      for j in range(len(q[0])):
        for k in range(len(q[0][j])):
          if q[0][j][k]==True:
            q[0][j][k]=255
          if q[0][j][k]==False:
            q[0][j][k]=0  
        
        q[0][j]=np.array(q[0][j])
    
      q[0] = np.array(q[0], dtype=np.uint8)
      data = q[0]
      img = Image.fromarray(data)
      img.save(os.path.join(self.PATH_im,str(self.filename)+'_instance_'+str(i)+'.png'))
      img.show()
  
      
