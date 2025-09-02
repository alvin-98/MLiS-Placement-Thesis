one_shot_example_1 = """INPUT: charge conditions for memory_gb and vm_family:

for Dsv5 the following charges apply:
  - if memory_gb in [1.0, 64.0), charge is $0.05 × memory_gb
  - if memory_gb in [64.0, 256.0), charge is $0.04 × memory_gb
  - if memory_gb in [256.0, 2048.0), charge is $0.03 × memory_gb
for Esv5 the following charges apply:
  - if memory_gb in [1.0, 64.0), charge is $0.06 × memory_gb
  - if memory_gb in [64.0, 256.0), charge is $0.05 × memory_gb
  - if memory_gb in [256.0, 2048.0), charge is $0.04 × memory_gb
for Fsv2 the following charges apply:
  - if memory_gb in [1.0, 64.0), charge is $0.07 × memory_gb
  - if memory_gb in [64.0, 256.0), charge is $0.06 × memory_gb
  - if memory_gb in [256.0, 2048.0), charge is $0.05 × memory_gb

OUTPUT:
<table border="1" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>Memory Range (GB)</th>
      <th>Dsv5</th>
      <th>Esv5</th>
      <th>Fsv2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1 – 63</td>
      <td>$0.05 per GB</td>
      <td>$0.06 per GB</td>
      <td>$0.07 per GB</td>
    </tr>
    <tr>
      <td>64 – 255</td>
      <td>$0.04 per GB</td>
      <td>$0.05 per GB</td>
      <td>$0.06 per GB</td>
    </tr>
    <tr>
      <td>256 – 2048</td>
      <td>$0.03 per GB</td>
      <td>$0.04 per GB</td>
      <td>$0.05 per GB</td>
    </tr>
  </tbody>
</table>
"""

one_shot_example_2 = """INPUT: charge conditions for gpu_count and gpu_type:

for T4 the following charges apply:
  - if gpu_count in [1, 2), charge is $0.35 × gpu_count
  - if gpu_count in [2, 4), charge is $0.32 × gpu_count
  - if gpu_count in [4, 8), charge is $0.30 × gpu_count
for A100 the following charges apply:
  - if gpu_count in [1, 2), charge is $1.20 × gpu_count
  - if gpu_count in [2, 4), charge is $1.10 × gpu_count
  - if gpu_count in [4, 8), charge is $1.00 × gpu_count
for MI300X the following charges apply:
  - if gpu_count in [1, 2), charge is $2.00 × gpu_count
  - if gpu_count in [2, 4), charge is $1.80 × gpu_count
  - if gpu_count in [4, 8), charge is $1.60 × gpu_count

OUTPUT:
<table border="1" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>GPU Count</th>
      <th>T4</th>
      <th>A100</th>
      <th>MI300X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>$0.35 per GPU / hour</td>
      <td>$1.20 per GPU / hour</td>
      <td>$2.00 per GPU / hour</td>
    </tr>
    <tr>
      <td>2 – 3</td>
      <td>$0.32 per GPU / hour</td>
      <td>$1.10 per GPU / hour</td>
      <td>$1.80 per GPU / hour</td>
    </tr>
    <tr>
      <td>4 – 7</td>
      <td>$0.30 per GPU / hour</td>
      <td>$1.00 per GPU / hour</td>
      <td>$1.60 per GPU / hour</td>
    </tr>
  </tbody>
</table>
"""

one_shot_example_3 = """INPUT: charge conditions for region and data_egress_gb:

for West US 2 the following charges apply:
  - charge is $0.01 per GB
for North Europe the following charges apply:
  - charge is $0.02 per GB
for East US the following charges apply:
  - charge is $0.03 per GB
for West Europe the following charges apply:
  - charge is $0.04 per GB
for UK West the following charges apply:
  - charge is $0.05 per GB
for UK South the following charges apply:
  - charge is $0.06 per GB

OUTPUT:
<table border="1" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>Geo Region</th>
      <th>Charge (USD per GB)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>West US 2</td>
      <td>$0.01</td>
    </tr>
    <tr>
      <td>North Europe</td>
      <td>$0.02</td>
    </tr>
    <tr>
      <td>East US</td>
      <td>$0.03</td>
    </tr>
    <tr>
      <td>West Europe</td>
      <td>$0.04</td>
    </tr>
    <tr>
      <td>UK West</td>
      <td>$0.05</td>
    </tr>
    <tr>
      <td>UK South</td>
      <td>$0.06</td>
    </tr>
  </tbody>
</table>
"""

