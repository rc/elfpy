ElfPy (Experimental lab fits in Python)
=======================================

A collection of Python modules with a command line interface to
(semi-)automatically evaluate and fit curves from experimental measurements of
mainly biological tissues.

The purpose of elfpy is to make the evaluation of mechanical measurements of
soft biological tissues easier and to enable a (semi-)automatic determination
of mechanical properties of these kinds of tissues.

Installation
------------

Install from sources::

  git clone https://github.com/rc/elfpy.git
  cd elfpy
  pip install .

Update existing git repository (deletes all uncommitted local changes!)::

  git fetch origin
  git reset --hard origin/master
  pip install .

Usage
-----

1. Convert data files to a suitable form using `elfpy-convert`.

   Run::

     elfpy-convert -h

   to get help.

2. Analyze the converted data using `elfpy-process`.

   Run::

     elfpy-process -h

   to get help and then::

     elfpy-process -l

   to see all available commands.

   Example command file::

     # Beginning of example command file.

     # Filters.
     smooth_strain
     smooth_stress
     select_cycle, -1
     get_ultimate_values

     -----

     # Plot commands.
     use_markers, 0

     plot_stress_strain, 1, 0, 'stress-strain'
     mark_ultimate_values, 1, 1

     -----

     # Save commands.
     save_ultimate_values
     save_figure, 1

     # End of example command file.
