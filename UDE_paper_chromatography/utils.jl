using LinearAlgebra
using BlockDiagonals

function make_OCFEM(n_elements::Int, n_phases::Int, n_components::Int; p_order = 4, xₘᵢₙ = 0.0, xₘₐₓ = 1.0)
"""
creates matrices H, A, B for discretization with orthogonal collocation on finite elements using cubic hermite polynomials

H are the polynomial evaluations at collocation points, B first derivative, C second derivative

"""
U = Array([0.2113248654, 0.7886751346]) #Collocation points  - zeros of shifted orthogonal Legendre polynomials
collocation_points = 2;
h = (xₘₐₓ -  xₘᵢₙ)/n_elements; # Finite elements size

#Initializing matrices
H = zeros(n_elements*collocation_points, p_order + 2*n_elements - 2);
A = zeros(n_elements*collocation_points, p_order + 2*n_elements - 2);
B = zeros(n_elements*collocation_points, p_order + 2*n_elements - 2);


function make_H(H::AbstractArray)
    l=1
    for j = 1:n_elements
        for k = 1:collocation_points
            for i = 1:p_order
                if i == 1
                    H[k-1 + l, i + 2*j - 2] = (1 +2*U[k])*(1 - U[k])^2
                
                elseif i == 2
                    H[k-1 + l, i + 2*j - 2] = U[k]*(1 - U[k])^2*h
                
                elseif i == 3
                    H[k-1 + l, i + 2*j - 2] = U[k]^2*(3 - 2*U[k])
                
                elseif i == 4
                    H[k-1 + l, i + 2*j - 2] = U[k]^2*(U[k] - 1)*h
                end
    
            end
        end
        l = l + 2
    end
    
    h1 = zeros(1,size(H)[2])
    h2 = zeros(1,size(H)[2])
    for i = 1:p_order
        if i == 1
            h1[i] = (1 + 2*0)*(1 - 0)^2
            h2[size(H)[2] - 4 + i] = (1 + 2*1)*(1 - 1)^2
        elseif i == 2
            h1[i] = 0*(1 - 0)^2*h
            h2[size(H)[2] - 4 + i] = 1*(1 - 1)^2*h
        elseif i == 3
            h1[i] = 0^2*(3 - 2*0)^2
            h2[size(H)[2] - 4 + i] = 1^2*(3 - 2*1)
        elseif i == 4
            h1[i] = 0^2*(0 - 1)*h
            h2[size(H)[2] - 4 + i] = 1^2*(1 - 1)*h
        end
    end
    
    H_aug = [h1; H; h2];   
    return H_aug 
end

function make_A(A::AbstractArray)
    l=1
    for j = 1:n_elements
        for k = 1:collocation_points
            for i = 1:p_order
                if i == 1
                    A[k-1 + l, i + 2*j - 2] = 6*U[k]^2 - 6*U[k]
                
                elseif i == 2
                    A[k-1 + l, i + 2*j - 2] = (1 - 4*U[k] + 3*U[k]^2)*h
                
                elseif i == 3
                    A[k-1 + l, i + 2*j - 2] = 6*U[k] - 6*U[k]^2
                
                elseif i == 4
                    A[k-1 + l, i + 2*j - 2] = (3*U[k]^2 - 2*U[k])*h
                end
    
            end
        end
        l = l + 2
    end
    
    a1 = zeros(1,size(A)[2])
    a2 = zeros(1,size(A)[2])
    for i = 1:p_order
        if i == 1
            a1[i] = 6*0^2 - 6*0
            a2[size(A)[2] - 4 + i] = 6*1^2 - 6*1
        elseif i == 2
            a1[i] = (1 - 4*0 + 3*0^2)*h
            a2[size(A)[2] - 4 + i] = (1 - 4*1 + 3*1^2)*h
        elseif i == 3
            a1[i] = 6*0 - 6*0^2
            a2[size(A)[2] - 4 + i] = 6*1 - 6*1^2
        elseif i == 4
            a1[i] = (3*0^2 - 2*0)*h
            a2[size(A)[2] - 4 + i] = (3*1^2 - 2*1)*h
        end
    end
    
    A_aug = [a1; A; a2];
    return A_aug
end
    

function make_B(B::AbstractArray)
    l=1
    for j = 1:n_elements
        for k = 1:collocation_points
            for i = 1:p_order
                if i == 1
                    B[k-1 + l, i + 2*j - 2] = 12*U[k] - 6
                
                elseif i == 2
                    B[k-1 + l, i + 2*j - 2] = (6*U[k] - 4)*h
                
                elseif i == 3
                    B[k-1 + l, i + 2*j - 2] = 6 - 12*U[k]
                
                elseif i == 4
                    B[k-1 + l, i + 2*j - 2] = (6*U[k] - 2)*h
                end
    
            end
        end
        l = l + 2
    end
    
    b1 = zeros(1,size(B)[2])
    b2 = zeros(1,size(B)[2])
    
    for i = 1:p_order
        if i == 1
            b1[i] = 12*0 - 6
            b2[size(B)[2] - 4 + i] = 12*1 - 6
        elseif i == 2
            b1[i] = (6*0 - 4)*h
            b2[size(B)[2] - 4 + i] = (6*1 - 4)*h
        elseif i == 3
            b1[i] = 6 - 12*0
            b2[size(B)[2] - 4 + i] = 6 - 12*1
        elseif i == 4
            b1[i] = (6*0 − 2)*h
            b2[size(B)[2] - 4 + i] = (6*1 - 2)*h
        end
    end
    
    B_aug = [b1; B; b2];
    return B_aug    
end

H_aug = make_H(H);
A_aug = make_A(A);
B_aug = make_B(B);

H_sparse = BlockDiagonal([H_aug for i = 1:n_components*n_phases]);
A_sparse = BlockDiagonal([A_aug for i = 1:n_components*n_phases]);
B_sparse = BlockDiagonal([B_aug for i = 1:n_components*n_phases]);

return H_sparse, A_sparse, B_sparse
end

function make_MM(n_elements::Int, n_phases::Int, n_components::Int; p_order = 4)
"""
make mass mass matrix:

stack these two matrices

|0 0 0 .... 0 0|                            |1 0 0 .... 0 0|  
|0 1 0 .....0 0|                            |0 1 0 .....0 0| -> solid phase
|0 0 1......0 0| - > liquid phase           |0 0 1......0 0|
|0 0 0......0 0|                            |0 0 0......0 1|

"""
n_variables = n_components*n_phases*(p_order + 2*n_elements-2)    

MM = Diagonal(zeros(n_variables, n_variables))
j = 0
    for i in 1:n_components
        #Internal node equations
        cl_idx = 2 + j
        cu_idx = p_order + 2*n_elements - 3 + j
    
        ql_idx = 2*(p_order + 2*n_elements - 2) + 2 + j
        qu_idx = p_order + 2*n_elements - 3 + 2*(p_order + 2*n_elements - 2) + j
    
        ql_idx2 = 2*(p_order + 2*n_elements - 2) + 2 + j - 1 #ql_idx - 1
        qu_idx2 = p_order + 2*n_elements - 3 + 2*(p_order + 2*n_elements - 2) + j + 1 #ql_idx + 1
    
        cbl_idx = j + 1
        cbu_idx = j + p_order + 2*n_elements - 2
        
    
        #Liquid phase residual
        MM[cl_idx:cu_idx, cl_idx:cu_idx] = Diagonal(ones(cu_idx - cl_idx + 1))
        #Solid phase residual
        MM[ql_idx2:qu_idx2, ql_idx2:qu_idx2] = Diagonal(ones(qu_idx2 - ql_idx2 + 1))
    
        j = j + p_order + 2*n_elements - 2
    end
    return MM
end


function make_MM_2(n_elements::Int, n_phases::Int, n_components::Int; p_order = 4)
    """
    make mass mass matrix:
    
    stack these two matrices
    
    |0 0 0 .... 0 0|                            |1 0 0 .... 0 0|  
    |0 1 0 .....0 0|                            |0 1 0 .....0 0| -> solid phase
    |0 0 1......0 0| - > liquid phase           |0 0 1......0 0|
    |0 0 0......0 0|                            |0 0 0......0 1|
    
    """
    n_variables = n_components*n_phases*(p_order + 2*n_elements-2)    
    
    MM = Diagonal(zeros(n_variables, n_variables))
    j = 0
        for i in 1:n_components
            #Internal node equations
            cl_idx = 2 + j
            cu_idx = p_order + 2*n_elements - 3 + j
        
            ql_idx = i*(p_order + 2*n_elements - 2) + 2 + j
            qu_idx = p_order + 2*n_elements - 3 + 2*(p_order + 2*n_elements - 2) + j
        
            ql_idx2 = i*(p_order + 2*n_elements - 2) + 2 + j - 1 #ql_idx - 1
            qu_idx2 = p_order + 2*n_elements - 3 + i*(p_order + 2*n_elements - 2) + j + 1 #ql_idx + 1
        
            cbl_idx = j + 1
            cbu_idx = j + p_order + 2*n_elements - 2
            
        
            #Liquid phase residual
            MM[cl_idx:cu_idx, cl_idx:cu_idx] = Diagonal(ones(cu_idx - cl_idx + 1))
            #Solid phase residual
            if n_phases > 1
                MM[ql_idx2:qu_idx2, ql_idx2:qu_idx2] = Diagonal(ones(qu_idx2 - ql_idx2 + 1))
            end
        
            j = j + p_order + 2*n_elements - 2
        end
        return MM
    end