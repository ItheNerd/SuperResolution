# SuperResolution
<section>
<h2>Steps to Implement</h2>
<ul>
<li>run <code>py -u make_x4plus_pickle.py</code></li>
<li>run <code>py -u enhanced_output.py</code> to see the image saved as a local file.</li>
<li>run <code>uvicorn main:app --reload</code> to run the server and access the API</li>
</ul>
<hr/>
<div>While running the server, send the input file as {"file": input_image}</div>
<div>Change the pre-trained weights used according to the use-case in the make_x4plus_pickle.py</div>
</section>

