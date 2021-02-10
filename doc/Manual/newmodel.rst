.. _newmodel:

Making your own Model
=========================
One of the key features of qubefit is the ability to generate your own models. The model can be simple or complex and can have any number of parameters that you want or need. The code will look for the model in the ``qfmodels.py`` file, so any models you wish to add needs to be added here, and they will be available for you to call using the methods defined in the tutorial and manual. The model should be structured as:

.. code::

   def your_model(**kwargs):

      # create a model with the parameters stored in the format:
      # kwargs['par']['name_of_parameter']
      Model = ...

      # you probably want to add these line last to your function
      # to convolve the model with the PSF/LSF
      if kwargs['convolve']:
         Model = convolve(Model, kwargs['kernel'])
   
   return Model

Here the model that is returned is a simple numpy array with dimension equal to the shape of the data cube. The parameters get passed in through \**kwargs. This last step is necessary because of the way *emcee* reads in the model. Any of the parameters that are defined in the model can be read in, in exactly the same way as with any of the pre-defined models. This is preferably done through a setup file (see :ref:`exfile`). For examples of models you can look at the pre-defined models in :ref:`qfmodels`. If you have created a model that might be useful for others, and wish to add it to the pre-defined list, please contact me, and I can add it.
